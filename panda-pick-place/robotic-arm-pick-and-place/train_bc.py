"""Train a CPU behavior-cloning baseline on the saved whole-episode split."""

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


def build_policy():
    return nn.Sequential(
        nn.Linear(25, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 8),
    )


def load_group(folder, records):
    observations, actions = [], []
    for record in records:
        path = (folder / record["file"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Episode path must stay within its batch folder.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Episode changed after splitting: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            obs, act = saved["observations"], saved["actions"]
            if obs.shape != (record["samples"], 25) or act.shape != (len(obs), 8):
                raise ValueError(f"Unexpected sample shape: {path.name}")
            if not bool(saved["expert_success"]):
                raise ValueError(f"Expected a successful expert episode: {path.name}")
            if not (np.isfinite(obs).all() and np.isfinite(act).all()):
                raise ValueError(f"Non-finite data: {path.name}")
            if not np.isin(act[:, 7], [0.0, 1.0]).all():
                raise ValueError(f"Expected binary gripper labels: {path.name}")
            observations.append(obs.copy())
            actions.append(act.copy())
    return np.concatenate(observations), np.concatenate(actions)


def training_normalizer(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    # Leave nearly constant dimensions unscaled instead of amplifying noise.
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def imitation_loss(outputs, targets):
    arm_loss = nn.functional.mse_loss(outputs[:, :7], targets[:, :7])
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(outputs[:, 7], targets[:, 7])
    return arm_loss + gripper_loss


@torch.no_grad()
def evaluate(model, observations, targets, arm_std):
    model.eval()
    total_loss, squared_error, correct, count = 0.0, 0.0, 0, len(observations)
    for start in range(0, count, 2048):
        x, y = observations[start:start + 2048], targets[start:start + 2048]
        outputs = model(x)
        total_loss += imitation_loss(outputs, y).item() * len(x)
        # Both angle arrays share their training-set mean, so it cancels here.
        squared_error += (((outputs[:, :7] - y[:, :7]) * arm_std) ** 2).sum().item()
        correct += ((outputs[:, 7] >= 0) == (y[:, 7] >= 0.5)).sum().item()
    return {
        "val_loss": total_loss / count,
        "val_arm_rmse_rad": (squared_error / (count * 7)) ** 0.5,
        "val_gripper_accuracy": correct / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--label", default="bc_baseline", help="Run label recorded in config")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or args.seed < 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive; seed must be non-negative.")

    split_path = args.split.resolve()
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    if split.get("schema") != "panda_green_bc_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected the saved whole-episode BC split.")
    train_records, val_records = split["train"], split["validation"]
    if not train_records or not val_records:
        raise ValueError("Both training and validation require complete episodes.")
    all_records = train_records + val_records
    if len({item["file"] for item in all_records}) != len(all_records) or len({item["seed"] for item in all_records}) != len(all_records):
        raise ValueError("Training and validation episode identities must be unique and disjoint.")
    if hashlib.sha256((split_path.parent / split["source_manifest"]).read_bytes()).hexdigest() != split["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after splitting.")

    train_obs, train_actions = load_group(split_path.parent, train_records)
    val_obs, val_actions = load_group(split_path.parent, val_records)
    if len(train_obs) != split["train_samples"] or len(val_obs) != split["validation_samples"]:
        raise ValueError("Loaded sample counts differ from the split.")

    obs_mean, obs_std = training_normalizer(train_obs)
    arm_mean, arm_std = training_normalizer(train_actions[:, :7])

    def prepare(observations, actions):
        x = torch.from_numpy(((observations - obs_mean) / obs_std).astype(np.float32))
        y = actions.astype(np.float32).copy()
        y[:, :7] = (y[:, :7] - arm_mean) / arm_std
        return x, torch.from_numpy(y)

    train_x, train_y = prepare(train_obs, train_actions)
    val_x, val_y = prepare(val_obs, val_actions)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = build_policy().to("cpu")
    initial_weight = model[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    arm_std_tensor = torch.from_numpy(arm_std)

    root = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"train_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "schema": "panda_green_bc_training_v1",
        "label": args.label,
        "device": "cpu",
        "architecture": "25 -> 128 ReLU -> 128 ReLU -> 8 (7 normalized angles, 1 gripper logit)",
        "loss": "mean arm normalized MSE + mean gripper BCEWithLogits",
        "normalization_source": "training episodes only",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed, "threads": args.threads,
        "split_path": str(split_path),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "train_seeds": [item["seed"] for item in train_records],
        "validation_seeds": [item["seed"] for item in val_records],
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    initial_metrics = evaluate(model, val_x, val_y, arm_std_tensor)
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    best_loss, best_epoch = float("inf"), 0

    def checkpoint(epoch, metrics):
        # Tensors and plain Python values allow loading with weights_only=True.
        return {
            "schema": "panda_green_bc_policy_v1",
            "model_state": model.state_dict(),
            "obs_mean": torch.from_numpy(obs_mean), "obs_std": torch.from_numpy(obs_std),
            "arm_mean": torch.from_numpy(arm_mean), "arm_std": arm_std_tensor,
            "epoch": epoch, "metrics": metrics, "config": config,
            "action_semantics": "7 absolute joint targets in rad; sigmoid(last logit) is gripper-open probability",
            "observation_schema": "panda_green_bc_v1, 25 values",
        }

    print(f"device = cpu; train = {len(train_x)}; validation = {len(val_x)}", flush=True)
    print(f"initial val_loss = {initial_metrics['val_loss']:.6f}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        training_total = 0.0
        for indices in order.split(args.batch_size):
            predictions = model(train_x[indices])
            loss = imitation_loss(predictions, train_y[indices])
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss; inspect this run before continuing.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_total += loss.item() * len(indices)
        metrics = evaluate(model, val_x, val_y, arm_std_tensor)
        row = {"epoch": epoch, "train_loss": training_total / len(train_x), **metrics}
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError("Non-finite epoch metrics.")
        history.append(row)
        if metrics["val_loss"] < best_loss:
            best_loss, best_epoch = metrics["val_loss"], epoch
            torch.save(checkpoint(epoch, metrics), output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch:02d}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"val_loss={metrics['val_loss']:.6f} arm_RMSE={metrics['val_arm_rmse_rad']:.6f} rad "
            f"gripper_acc={metrics['val_gripper_accuracy']:.4f}", flush=True,
        )

    torch.save(checkpoint(args.epochs, metrics), output / "last.pt")
    weight_change = (model[0].weight.detach() - initial_weight).abs().max().item()
    if weight_change == 0:
        raise RuntimeError("Network weights did not change.")
    summary = {
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "initial_val_loss": initial_metrics["val_loss"],
        "elapsed_seconds": time.perf_counter() - start,
        "first_layer_max_weight_change": weight_change,
        "closed_loop_policy_tested": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest epoch = {best_epoch}; best val_loss = {best_loss:.6f}")
    print(f"weights updated; elapsed = {summary['elapsed_seconds']:.1f} s")
    print(f"saved = {output}")
    print("Offline imitation training completed. Gripper accuracy is not grasp success rate.")


if __name__ == "__main__":
    main()
