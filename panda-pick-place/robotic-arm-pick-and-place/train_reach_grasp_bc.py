"""Train a bounded 10-input/4-output reach-and-grasp BC policy on CPU."""

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

from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask


PREVIOUS_OPEN_INDEX = OBSERVATION_NAMES.index("previous_gripper_open")


def build_reach_grasp_policy():
    return nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 4),
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
            if metadata.get("schema") != "reach_grasp_episode_v1":
                raise ValueError("Expected the 10-input/4-output reach-and-grasp dataset.")
            expected_steps = int(round(ReachGraspTask.HORIZON / ReachGraspTask.CONTROL_DT))
            if obs.shape != (expected_steps, 10) or act.shape != (expected_steps, 4) or record["samples"] != expected_steps:
                raise ValueError("Expected a complete episode with observations (400,10) and actions (400,4).")
            if not (np.isfinite(obs).all() and np.isfinite(act).all()):
                raise ValueError("Non-finite recorded data.")
            if np.abs(act[:, :3]).max() > 1 + 1e-10 or not np.isin(act[:, 3], [0., 1.]).all():
                raise ValueError("Expected bounded XYZ increments and binary gripper commands.")
            if not (bool(saved["expert_success"]) and record["expert_success"] and metadata["expert_success"]):
                raise ValueError("This baseline requires successful expert demonstrations.")
            if not metadata["completed"] or metadata["stop_reason"] != "horizon_reached":
                raise ValueError("Incomplete expert episode.")
            if metadata["seed"] != record["seed"] or metadata["observation_names"] != OBSERVATION_NAMES:
                raise ValueError("Episode identity or observation schema differs.")
            if metadata["observation_scales"] != OBSERVATION_SCALES.tolist():
                raise ValueError("Observation scales differ from collection.")
            if (metadata["max_delta_xyz_m"] != ReachGraspTask.MAX_DELTA_XYZ
                    or metadata["control_dt_s"] != ReachGraspTask.CONTROL_DT
                    or metadata["horizon_s"] != ReachGraspTask.HORIZON):
                raise ValueError("Task execution contract changed since collection.")
            if metadata["setup_reference_m"] != ReachGraspTask.START_POSITION.tolist():
                raise ValueError("Fixed starting reference changed since collection.")
            if metadata["reference_bounds_m"] != [ReachGraspTask.REFERENCE_LOWER.tolist(), ReachGraspTask.REFERENCE_UPPER.tolist()]:
                raise ValueError("Reference workspace changed since collection.")
            if first_metadata is None:
                first_metadata = metadata
                for name, digest in metadata["source_sha256"].items():
                    source = (root / name).resolve()
                    if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Task source changed since collection: {name}")
            elif metadata["source_sha256"] != first_metadata["source_sha256"]:
                raise ValueError("Mixed task versions in dataset.")
            observations.append(obs.copy())
            bounded = act.copy()
            bounded[:, :3] = np.clip(bounded[:, :3], -1., 1.)  # Remove floating-point boundary roundoff.
            actions.append(bounded)
    return np.concatenate(observations), np.concatenate(actions), first_metadata


def imitation_loss(outputs, labels, previous_open, switch_weight):
    xyz_loss = nn.functional.mse_loss(outputs[:, :3].tanh(), labels[:, :3])
    # One close transition per episode: copying previous_open alone scores 99.75%.
    switches = (labels[:, 3] - previous_open).abs() > .5
    weights = 1. + switches.float() * (switch_weight - 1.)
    errors = nn.functional.binary_cross_entropy_with_logits(outputs[:, 3], labels[:, 3], reduction="none")
    return xyz_loss + (weights * errors).sum() / weights.sum()


@torch.no_grad()
def evaluate(policy, observations, labels, switch_weight):
    policy.eval()
    outputs = policy(observations)
    errors_mm = (outputs[:, :3].tanh() - labels[:, :3]) * ReachGraspTask.MAX_DELTA_XYZ * 1000
    axis_rmse = errors_mm.square().mean(dim=0).sqrt()
    switches = (labels[:, 3] - observations[:, PREVIOUS_OPEN_INDEX]).abs() > .5
    correct = (outputs[:, 3] >= 0) == (labels[:, 3] >= .5)
    active_xy = labels[:, :2].abs().amax(dim=1) > 1e-6
    active_xy_rmse = errors_mm[active_xy, :2].square().sum(dim=1).mean().sqrt().item() if active_xy.any().item() else 0.
    return {
        "val_loss": imitation_loss(outputs, labels, observations[:, PREVIOUS_OPEN_INDEX], switch_weight).item(),
        "val_x_step_rmse_mm": axis_rmse[0].item(), "val_y_step_rmse_mm": axis_rmse[1].item(),
        "val_z_step_rmse_mm": axis_rmse[2].item(),
        "val_xyz_step_rmse_mm": errors_mm.square().sum(dim=1).mean().sqrt().item(),
        "val_active_xy_step_rmse_mm": active_xy_rmse, "val_active_xy_samples": int(active_xy.sum().item()),
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
    parser.add_argument("--switch-weight", type=float, default=50.)
    parser.add_argument("--label", default="reach_grasp_bc")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Need positive training settings and non-negative seed.")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("--switch-weight must be finite and at least 1.")
    root = Path(__file__).resolve().parent
    split_path = args.split.resolve()
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    if split.get("schema") != "reach_grasp_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected a reach-and-grasp whole-episode split.")
    if not split["train"] or not split["validation"]:
        raise ValueError("Both training and validation episodes are required.")
    records = split["train"] + split["validation"]
    if len({record["seed"] for record in records}) != len(records) or len({record["file"] for record in records}) != len(records):
        raise ValueError("Episode identities must be unique and disjoint between splits.")
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
    policy = build_reach_grasp_policy().to("cpu")
    initial_weight = policy[0].weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    config = {
        "schema": "reach_grasp_training_v1", "label": args.label, "device": "cpu",
        "architecture": "10 -> 64 ReLU -> 64 ReLU -> 4",
        "outputs": "tanh(first three logits) are normalized delta-XYZ; sigmoid(last logit) is gripper open probability",
        "loss": "bounded XYZ action MSE + switch-weighted gripper BCEWithLogits", "switch_weight": args.switch_weight,
        "observation_names": OBSERVATION_NAMES, "normalization": "fixed physical scales; no validation-fitted statistics",
        "metric_units": "RMSE in mm measures reference-increment prediction, not actual end-effector tracking",
        "task_source_sha256": metadata["source_sha256"], "setup_reference_m": metadata["setup_reference_m"],
        "reference_bounds_m": metadata["reference_bounds_m"],
        "control_dt_s": metadata["control_dt_s"], "physics_dt_s": metadata["physics_dt_s"],
        "max_delta_xyz_m": metadata["max_delta_xyz_m"], "horizon_s": metadata["horizon_s"],
        "split_path": str(split_path), "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "train_seeds": [record["seed"] for record in split["train"]],
        "validation_seeds": [record["seed"] for record in split["validation"]],
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed, "threads": args.threads,
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "mujoco": metadata["mujoco"], "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_train_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    initial_metrics = evaluate(policy, val_x, val_y, args.switch_weight)
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    best_loss, best_epoch = float("inf"), 0

    def checkpoint(epoch, metrics):
        return {"schema": "reach_grasp_policy_v1", "model_state": policy.state_dict(),
                "observation_scales": torch.from_numpy(OBSERVATION_SCALES.copy()),
                "epoch": epoch, "metrics": metrics, "config": config}

    print(f"device = cpu; train = {len(train_x)}; validation = {len(val_x)}", flush=True)
    print(f"initial val_loss = {initial_metrics['val_loss']:.6f}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total = 0.
        for indices in torch.randperm(len(train_x)).split(args.batch_size):
            outputs = policy(train_x[indices])
            loss = imitation_loss(outputs, train_y[indices], train_x[indices, PREVIOUS_OPEN_INDEX], args.switch_weight)
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
            axes = ",".join(f"{metrics[f'val_{axis}_step_rmse_mm']:.4f}" for axis in "xyz")
            print(f"epoch {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.6f} val_loss={metrics['val_loss']:.6f} "
                  f"XYZ_step_RMSE_mm=[{axes}] gripper_acc={metrics['val_gripper_accuracy']:.4f} "
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
    print("Offline reach-and-grasp imitation only; actual grasping requires a closed-loop test.")


if __name__ == "__main__":
    main()
