"""Train a bounded six-input/two-output CPU policy for vertical grasping."""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, VerticalGraspTask


def build_grasp_policy():
    return nn.Sequential(
        nn.Linear(6, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def load_episodes(folder, records, root):
    observations, actions = [], []
    first_metadata = None
    for record in records:
        path = (folder / record["file"]).resolve()
        if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Episode path or hash mismatch: {record['file']}")
        with np.load(path, allow_pickle=False) as saved:
            obs, act = saved["observations"], saved["actions"]
            metadata = json.loads(str(saved["metadata_json"]))
            if metadata.get("schema") != "vertical_grasp_episode_v1":
                raise ValueError("Expected the new vertical grasp dataset, not the old joint-target dataset.")
            if obs.shape != (record["samples"], 6) or act.shape != (len(obs), 2):
                raise ValueError("Expected observations (N, 6) and actions (N, 2).")
            if not (np.isfinite(obs).all() and np.isfinite(act).all()):
                raise ValueError("Non-finite recorded data.")
            if np.abs(act[:, 0]).max() > 1 + 1e-10 or not np.isin(act[:, 1], [0., 1.]).all():
                raise ValueError("Unexpected bounded height action or binary gripper label.")
            if not bool(saved["expert_success"]) or not record["grasp_success"]:
                raise ValueError("This baseline requires successful expert demonstrations.")
            if metadata["seed"] != record["seed"] or metadata["observation_names"] != OBSERVATION_NAMES:
                raise ValueError("Episode identity or observation schema differs.")
            if metadata["observation_scales"] != OBSERVATION_SCALES.tolist():
                raise ValueError("Observation scales differ from the collected task.")
            if metadata["max_delta_z_m"] != VerticalGraspTask.MAX_DELTA_Z or metadata["control_dt_s"] != VerticalGraspTask.CONTROL_DT:
                raise ValueError("Action execution contract changed after data collection.")
            if first_metadata is None:
                first_metadata = metadata
                for name, digest in metadata["source_sha256"].items():
                    source = (root / name).resolve()
                    if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Task source changed since collection: {name}")
            elif metadata["source_sha256"] != first_metadata["source_sha256"]:
                raise ValueError("Mixed source versions in dataset.")
            observations.append(obs.copy())
            clipped = act.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], -1., 1.)  # Remove roundoff at +/-1.
            actions.append(clipped)
    return np.concatenate(observations), np.concatenate(actions), first_metadata


def imitation_loss(outputs, labels, previous_open, switch_weight):
    height_loss = nn.functional.mse_loss(outputs[:, 0].tanh(), labels[:, 0])
    # A demo has just one close transition. Copying the previous command can
    # score 99.67% per frame while never initiating a grasp in closed loop.
    switches = (labels[:, 1] - previous_open).abs() > 0.5
    weights = 1.0 + switches.float() * (switch_weight - 1.0)
    gripper_errors = nn.functional.binary_cross_entropy_with_logits(outputs[:, 1], labels[:, 1], reduction="none")
    gripper_loss = (weights * gripper_errors).sum() / weights.sum()
    return height_loss + gripper_loss


@torch.no_grad()
def evaluate(policy, observations, labels, switch_weight):
    policy.eval()
    outputs = policy(observations)
    height_error = outputs[:, 0].tanh() - labels[:, 0]
    switches = (labels[:, 1] - observations[:, 3]).abs() > 0.5
    correct = (outputs[:, 1] >= 0) == (labels[:, 1] >= .5)
    return {
        "val_loss": imitation_loss(outputs, labels, observations[:, 3], switch_weight).item(),
        "val_step_rmse_mm": height_error.square().mean().sqrt().item() * VerticalGraspTask.MAX_DELTA_Z * 1000,
        "val_gripper_accuracy": correct.float().mean().item(),
        "val_gripper_switch_accuracy": float(correct[switches].sum().item() / max(int(switches.sum().item()), 1)),
        "val_switch_samples": int(switches.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--switch-weight", type=float, default=50.0, help="Extra weight for changing the previous gripper command")
    parser.add_argument("--label", default="vertical_grasp_bc")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Need positive training settings and a non-negative seed.")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("--switch-weight must be finite and at least 1.")
    root = Path(__file__).resolve().parent
    split_path = args.split.resolve()
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    if split.get("schema") != "vertical_grasp_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected a vertical grasp whole-episode split.")
    records = split["train"] + split["validation"]
    if not split["train"] or not split["validation"]:
        raise ValueError("Both training and validation episodes are required.")
    if len({record["file"] for record in records}) != len(records) or len({record["seed"] for record in records}) != len(records):
        raise ValueError("Episode identities must be unique and disjoint across splits.")
    train_obs, train_actions, metadata = load_episodes(split_path.parent, split["train"], root)
    val_obs, val_actions, val_metadata = load_episodes(split_path.parent, split["validation"], root)
    if metadata["source_sha256"] != val_metadata["source_sha256"]:
        raise ValueError("Training and validation task versions differ.")
    train_x = torch.from_numpy((train_obs / OBSERVATION_SCALES).astype(np.float32))
    val_x = torch.from_numpy((val_obs / OBSERVATION_SCALES).astype(np.float32))
    train_y = torch.from_numpy(train_actions.astype(np.float32))
    val_y = torch.from_numpy(val_actions.astype(np.float32))
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    policy = build_grasp_policy().to("cpu")
    initial_weight = policy[0].weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    config = {
        "schema": "vertical_grasp_training_v1", "label": args.label, "device": "cpu",
        "architecture": "6 -> 64 ReLU -> 64 ReLU -> 2",
        "outputs": "tanh(height logit) is normalized delta-z; sigmoid(gripper logit) is open probability",
        "loss": "bounded height-action MSE + switch-weighted gripper BCEWithLogits",
        "switch_weight": args.switch_weight,
        "observation_names": OBSERVATION_NAMES,
        "normalization": "fixed physical scales; no validation-fitted statistics",
        "task_source_sha256": metadata["source_sha256"],
        "control_dt_s": metadata["control_dt_s"], "physics_dt_s": metadata["physics_dt_s"],
        "max_delta_z_m": metadata["max_delta_z_m"], "horizon_s": metadata["horizon_s"],
        "split_path": str(split_path), "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "train_seeds": [record["seed"] for record in split["train"]],
        "validation_seeds": [record["seed"] for record in split["validation"]],
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "threads": args.threads,
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"grasp_train_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    initial_metrics = evaluate(policy, val_x, val_y, args.switch_weight)
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    best_loss, best_epoch = float("inf"), 0

    def checkpoint(epoch, metrics):
        return {"schema": "vertical_grasp_policy_v1", "model_state": policy.state_dict(),
                "observation_scales": torch.from_numpy(OBSERVATION_SCALES.copy()),
                "epoch": epoch, "metrics": metrics, "config": config}

    print(f"device = cpu; train = {len(train_x)}; validation = {len(val_x)}", flush=True)
    print(f"initial val_loss = {initial_metrics['val_loss']:.6f}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total = 0.0
        for indices in torch.randperm(len(train_x)).split(args.batch_size):
            outputs = policy(train_x[indices])
            loss = imitation_loss(outputs, train_y[indices], train_x[indices, 3], args.switch_weight)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        metrics = evaluate(policy, val_x, val_y, args.switch_weight)
        row = {"epoch": epoch, "train_loss": total / len(train_x), **metrics}
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError("Non-finite validation metrics.")
        history.append(row)
        if metrics["val_loss"] < best_loss:
            best_loss, best_epoch = metrics["val_loss"], epoch
            torch.save(checkpoint(epoch, metrics), output / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.6f} "
                  f"val_loss={metrics['val_loss']:.6f} step_RMSE={metrics['val_step_rmse_mm']:.4f} mm "
                  f"gripper_acc={metrics['val_gripper_accuracy']:.4f} "
                  f"switch_acc={metrics['val_gripper_switch_accuracy']:.4f} (n={metrics['val_switch_samples']})", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(checkpoint(args.epochs, metrics), output / "last.pt")
    change = float((policy[0].weight.detach() - initial_weight).abs().max().item())
    if change == 0:
        raise RuntimeError("Weights did not update.")
    summary = {"initial_val_loss": initial_metrics["val_loss"], "best_epoch": best_epoch,
               "best_val_loss": best_loss, "elapsed_seconds": time.perf_counter() - start,
               "first_layer_max_weight_change": change, "closed_loop_tested": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest epoch = {best_epoch}; best val_loss = {best_loss:.6f}")
    print(f"weights updated; elapsed = {summary['elapsed_seconds']:.1f} s")
    print(f"saved = {output}")
    print("Offline grasp-action imitation only; actual grasping still requires a closed-loop test.")


if __name__ == "__main__":
    main()
