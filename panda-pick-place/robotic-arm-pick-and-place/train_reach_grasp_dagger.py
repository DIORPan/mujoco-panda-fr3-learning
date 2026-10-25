"""Fine-tune the BC network on original demos plus closed-gripper on-policy labels."""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask
from rollout_reach_grasp_bc import load_policy
from train_reach_grasp_bc import PREVIOUS_OPEN_INDEX, evaluate, imitation_loss, load_episodes


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dagger_labels(folder, records, manifest, root):
    observations, actions = [], []
    for record in records:
        path = (folder / record["file"]).resolve()
        if not path.is_relative_to(folder) or digest(path) != record["sha256"]:
            raise ValueError(f"Label episode path or hash mismatch: {record['file']}")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            if metadata.get("schema") != "reach_grasp_dagger_episode_v1":
                raise ValueError("Expected an on-policy expert-label episode.")
            if metadata["seed"] != record["seed"] or not metadata["collection_complete"] or not record["collection_complete"]:
                raise ValueError("Incomplete or mismatched on-policy episode.")
            if metadata["behavior_checkpoint_sha256"] != manifest["behavior_checkpoint_sha256"]:
                raise ValueError("Labels came from a different behavior policy.")
            if metadata["expert_actions_executed"] or not metadata["learner_executed_all_task_actions"]:
                raise ValueError("Expected learner-only action execution during collection.")
            if metadata["observation_names"] != OBSERVATION_NAMES or metadata["observation_scales"] != OBSERVATION_SCALES.tolist():
                raise ValueError("Label observation contract differs.")
            if (metadata["control_dt_s"] != ReachGraspTask.CONTROL_DT or metadata["max_delta_xyz_m"] != ReachGraspTask.MAX_DELTA_XYZ
                    or metadata["horizon_s"] != ReachGraspTask.HORIZON):
                raise ValueError("On-policy execution contract differs.")
            all_obs = saved["observations"]
            obs, act, steps = saved["expert_observations"], saved["expert_actions"], saved["expert_label_steps"]
            expected_steps = int(round(ReachGraspTask.HORIZON / ReachGraspTask.CONTROL_DT))
            if all_obs.shape != (expected_steps, 10) or len(steps) != record["labelled_states"] or len(steps) < 1:
                raise ValueError("Unexpected rollout or expert-label count.")
            if obs.shape != (len(steps), 10) or act.shape != (len(steps), 4):
                raise ValueError("Expected expert labels paired with ten-dimensional visited states.")
            if not (np.isfinite(all_obs).all() and np.isfinite(obs).all() and np.isfinite(act).all()):
                raise ValueError("Non-finite on-policy labels.")
            np.testing.assert_array_equal(steps, np.flatnonzero(all_obs[:, PREVIOUS_OPEN_INDEX] < .5))
            np.testing.assert_array_equal(obs, all_obs[steps])
            if np.abs(act[:, :3]).max() > 1 + 1e-10 or not np.isin(act[:, 3], [0., 1.]).all():
                raise ValueError("Unexpected expert action bounds.")
            # Failed learner rollouts are deliberately retained when their labels are valid.
            observations.append(obs.copy())
            bounded = act.copy()
            bounded[:, :3] = np.clip(bounded[:, :3], -1., 1.)
            actions.append(bounded)
    return np.concatenate(observations), np.concatenate(actions)


def tensor_inputs(obs):
    return torch.from_numpy((obs / OBSERVATION_SCALES).astype(np.float32))


def group_metrics(policy, bc_x, bc_y, dagger_x, dagger_y, switch_weight):
    bc = evaluate(policy, bc_x, bc_y, switch_weight)
    dagger = evaluate(policy, dagger_x, dagger_y, switch_weight)
    return {"selection_val_loss": .5 * (bc["val_loss"] + dagger["val_loss"]),
            **{f"bc_{key}": value for key, value in bc.items()},
            **{f"dagger_{key}": value for key, value in dagger.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dagger-split", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--switch-weight", type=float, default=50.)
    parser.add_argument("--label", default="closed_gripper_dataset_aggregation_round1")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Need positive training settings and non-negative seed.")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("--switch-weight must be finite and at least 1.")
    root = Path(__file__).resolve().parent
    label_split_path = args.dagger_split.resolve()
    label_split = read_json(label_split_path)
    manifest_path = label_split_path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    if label_split.get("schema") != "reach_grasp_dagger_split_v1" or label_split.get("unit") != "whole_episode":
        raise ValueError("Expected a whole-episode on-policy label split.")
    if manifest.get("schema") != "reach_grasp_dagger_batch_v1" or not manifest["completed_batch"]:
        raise ValueError("Expected a completed label-collection manifest.")
    if manifest["network_updated"] or manifest["expert_labels_executed"] or not manifest["checkpoint_file_unchanged"]:
        raise ValueError("Label-collection provenance is inconsistent.")
    checkpoint_path = Path(manifest["behavior_checkpoint"])
    policy, scales, base_checkpoint, bc_split, base_hash = load_policy(checkpoint_path, root)
    if base_hash != manifest["behavior_checkpoint_sha256"]:
        raise ValueError("Behavior checkpoint differs from collection.")
    base_config = base_checkpoint["config"]
    if digest(root / "train_reach_grasp_bc.py") != base_config["trainer_sha256"]:
        raise ValueError("Original model implementation changed since BC training.")
    bc_split_path = Path(base_config["split_path"])
    if str(bc_split_path) != manifest["original_split_path"] or digest(bc_split_path) != manifest["original_split_sha256"]:
        raise ValueError("Original expert split differs from label collection.")
    if manifest["source_sha256"] != base_config["task_source_sha256"]:
        raise ValueError("Label and original task source versions differ.")
    all_records = bc_split["train"] + bc_split["validation"] + label_split["train"] + label_split["validation"]
    if any(not split[group] for split in (bc_split, label_split) for group in ("train", "validation")):
        raise ValueError("Every data source needs separate training and validation episodes.")
    if len({r["seed"] for r in all_records}) != len(all_records):
        raise ValueError("Episode seeds overlap across data sources or training/validation groups.")
    manifest_records = {record["seed"]: record for record in manifest["episodes"]}
    if {r["seed"] for r in label_split["train"] + label_split["validation"]} != set(manifest_records):
        raise ValueError("Label split must retain every collected episode.")
    for r in label_split["train"] + label_split["validation"]:
        if r != manifest_records[r["seed"]]:
            raise ValueError("Label split record differs from its manifest.")
    bc_train_obs, bc_train_actions, _ = load_episodes(bc_split_path.parent, bc_split["train"], root)
    bc_val_obs, bc_val_actions, _ = load_episodes(bc_split_path.parent, bc_split["validation"], root)
    label_train_obs, label_train_actions = load_dagger_labels(label_split_path.parent, label_split["train"], manifest, root)
    label_val_obs, label_val_actions = load_dagger_labels(label_split_path.parent, label_split["validation"], manifest, root)
    train_x = tensor_inputs(np.concatenate([bc_train_obs, label_train_obs]))
    train_y = torch.from_numpy(np.concatenate([bc_train_actions, label_train_actions]).astype(np.float32))
    bc_val_x, label_val_x = tensor_inputs(bc_val_obs), tensor_inputs(label_val_obs)
    bc_val_y = torch.from_numpy(bc_val_actions.astype(np.float32))
    label_val_y = torch.from_numpy(label_val_actions.astype(np.float32))
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    policy.requires_grad_(True)
    initial_weight = policy[0].weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_dagger_train_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    aggregate_split = {"schema": "reach_grasp_aggregate_split_v1", "unit": "whole_episode"}
    for group in ("train", "validation"):
        aggregate_split[group] = [
            {"kind": kind, "path": str((folder / record["file"]).resolve()), "seed": record["seed"],
             "sha256": record["sha256"], "samples": record[count_key]}
            for kind, folder, split, count_key in (
                ("bc_expert", bc_split_path.parent, bc_split, "samples"),
                ("closed_gripper_on_policy_labels", label_split_path.parent, label_split, "labelled_states"))
            for record in split[group]
        ]
    aggregate_path = output / "aggregate_split.json"
    aggregate_path.write_text(json.dumps(aggregate_split, indent=2), encoding="utf-8")
    failed_train_episodes = sum(not r["behavior_policy_success"] for r in label_split["train"])
    config = {
        **base_config, "schema": "reach_grasp_dagger_training_v1", "label": args.label,
        "initialization": "fine-tune original BC best checkpoint; fresh Adam optimizer",
        "initial_checkpoint": str(checkpoint_path), "initial_checkpoint_sha256": base_hash,
        "method": "one targeted dataset-aggregation round using closed-gripper learner-visited states",
        "split_path": str(aggregate_path), "split_sha256": digest(aggregate_path),
        "data_sources": {"bc_split": {"path": str(bc_split_path), "sha256": digest(bc_split_path)},
                         "label_split": {"path": str(label_split_path), "sha256": digest(label_split_path)},
                         "label_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)}},
        "train_seeds": [r["seed"] for r in aggregate_split["train"]],
        "validation_seeds": [r["seed"] for r in aggregate_split["validation"]],
        "training_samples": {"bc": len(bc_train_obs), "on_policy_labels": len(label_train_obs)},
        "validation_samples": {"bc": len(bc_val_obs), "on_policy_labels": len(label_val_obs)},
        "failed_learner_training_episodes_retained": failed_train_episodes,
        "validation_selection": "equal-weight mean of BC validation loss and on-policy-label validation loss",
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "threads": args.threads, "switch_weight": args.switch_weight,
        "model_builder_sha256": base_config["trainer_sha256"], "trainer_sha256": digest(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    initial_metrics = group_metrics(policy, bc_val_x, bc_val_y, label_val_x, label_val_y, args.switch_weight)
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    best_loss, best_epoch = float("inf"), 0

    def checkpoint(epoch, metrics):
        return {"schema": "reach_grasp_dagger_policy_v1", "model_state": policy.state_dict(),
                "observation_scales": torch.from_numpy(scales.copy()), "epoch": epoch,
                "metrics": metrics, "config": config}

    print(f"device = cpu; train = {len(train_x)} (BC {len(bc_train_obs)} + labels {len(label_train_obs)})", flush=True)
    print(f"validation = BC {len(bc_val_obs)} + labels {len(label_val_obs)}; failed learner train episodes retained = {failed_train_episodes}", flush=True)
    print(f"initial selection_val_loss = {initial_metrics['selection_val_loss']:.6f}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total = 0.
        for indices in torch.randperm(len(train_x)).split(args.batch_size):
            loss = imitation_loss(policy(train_x[indices]), train_y[indices], train_x[indices, PREVIOUS_OPEN_INDEX], args.switch_weight)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite aggregation training loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        metrics = group_metrics(policy, bc_val_x, bc_val_y, label_val_x, label_val_y, args.switch_weight)
        row = {"epoch": epoch, "train_loss": total / len(train_x), **metrics}
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError("Non-finite validation metrics.")
        history.append(row)
        if metrics["selection_val_loss"] < best_loss:
            best_loss, best_epoch = metrics["selection_val_loss"], epoch
            torch.save(checkpoint(epoch, metrics), output / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.6f} val_loss={metrics['selection_val_loss']:.6f} "
                  f"BC_XYZ_RMSE={metrics['bc_val_xyz_step_rmse_mm']:.4f} mm "
                  f"label_XYZ_RMSE={metrics['dagger_val_xyz_step_rmse_mm']:.4f} mm "
                  f"BC_switch_acc={metrics['bc_val_gripper_switch_accuracy']:.4f}", flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(checkpoint(args.epochs, metrics), output / "last.pt")
    change = float((policy[0].weight.detach() - initial_weight).abs().max().item())
    if change == 0 or digest(checkpoint_path) != base_hash:
        raise RuntimeError("Weights did not update or original checkpoint changed.")
    summary = {"initial_selection_val_loss": initial_metrics["selection_val_loss"], "best_epoch": best_epoch,
               "best_selection_val_loss": best_loss, "elapsed_seconds": time.perf_counter() - start,
               "first_layer_max_weight_change": change, "original_checkpoint_unchanged": True, "closed_loop_tested": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest epoch = {best_epoch}; best val_loss = {best_loss:.6f}")
    print(f"weights updated; elapsed = {summary['elapsed_seconds']:.1f} s")
    print(f"saved = {output}")
    print("Supervised aggregation training only. Improved closed-loop success has not been established.")


if __name__ == "__main__":
    main()
