"""Collect on-policy expert labels after gripper closure; no network updates.

This is a targeted dataset-aggregation step, not a full DAgger benchmark.
The frozen learner executes every task action. Expert labels are recorded
at its visited closed-gripper states and never applied to the simulation.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch

from collect_reach_grasp_dataset import save_json
from reach_grasp_task import OBSERVATION_NAMES, ReachGraspTask, expert_action
from rollout_reach_grasp_bc import load_policy


@torch.inference_mode()
def collect_episode(model, policy, scales, seed, checkpoint_hash):
    task = ReachGraspTask(model)
    observation = task.reset(seed)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, task.data, initial, spec)
    initial_ctrl = task.data.ctrl.copy()
    observations, requested, applied, references = [], [], [], []
    expert_observations, expert_actions, expert_steps = [], [], []
    reason = "horizon_reached"
    for _ in range(int(round(task.HORIZON / task.CONTROL_DT))):
        inputs = torch.from_numpy((observation / scales).astype(np.float32))
        outputs = policy(inputs)
        if outputs.shape != (4,) or not torch.isfinite(outputs).all().item():
            reason = "invalid_policy_prediction"
            break
        learner_action = np.concatenate([outputs[:3].tanh().numpy(), [outputs[3].sigmoid().item()]])
        reference = task.reference_position.copy()
        label = None
        # Selection is observable command state, never rollout success/failure.
        if observation[5] < .5:
            label = expert_action(observation)
            bounded_reference = np.clip(reference + np.clip(label[:3], -1, 1) * task.MAX_DELTA_XYZ,
                                        task.REFERENCE_LOWER, task.REFERENCE_UPPER)
            label = np.concatenate([(bounded_reference - reference) / task.MAX_DELTA_XYZ,
                                    [float(label[3] >= .5)]])
        try:
            next_observation, executed = task.step(learner_action)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            reason = f"execution_error: {error}"
            break
        if label is not None:
            expert_steps.append(len(observations))
            expert_observations.append(observation.copy())
            expert_actions.append(label)
        observations.append(observation.copy())
        requested.append(learner_action)
        applied.append(executed)
        references.append(reference)
        observation = next_observation
    result = task.result()
    arrays = {
        "observations": np.asarray(observations).reshape(-1, 10),
        "learner_requested_actions": np.asarray(requested).reshape(-1, 4),
        "learner_applied_actions": np.asarray(applied).reshape(-1, 4),
        "reference_positions": np.asarray(references).reshape(-1, 3),
        "expert_observations": np.asarray(expert_observations).reshape(-1, 10),
        "expert_actions": np.asarray(expert_actions).reshape(-1, 4),
        "expert_label_steps": np.asarray(expert_steps, dtype=np.int64),
        "initial_state": initial, "initial_ctrl": initial_ctrl,
        "final_observation": observation,
    }
    finite = bool(all(np.isfinite(value).all() for value in arrays.values())
                  and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    complete = bool(reason == "horizon_reached" and result["completed"] and finite)
    metadata = {
        "schema": "reach_grasp_dagger_episode_v1", "seed": seed,
        "behavior_checkpoint_sha256": checkpoint_hash,
        "label_rule": "expert labels at learner-visited states where previous_gripper_open < 0.5",
        "expert_actions_executed": False, "learner_executed_all_task_actions": True,
        "observation_names": OBSERVATION_NAMES, "observation_scales": scales.tolist(),
        "initial_state_spec": spec, "initial_cube_xy_m": task.requested_cube_xy.tolist(),
        "setup_reference_m": task.START_POSITION.tolist(), "control_dt_s": task.CONTROL_DT,
        "max_delta_xyz_m": task.MAX_DELTA_XYZ, "horizon_s": task.HORIZON,
        "stop_reason": reason, "collection_complete": complete, "finite_data": finite,
        "labelled_states": len(expert_steps), "rollout_steps": len(observations), **result,
        "behavior_policy_success": bool(complete and result["grasp_success"]),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--label", default="closed_gripper_on_policy_labels_round1")
    args = parser.parse_args()
    if args.episodes < 2 or args.seed < 0 or args.split_seed < 0 or not 1 <= args.validation_episodes < args.episodes:
        parser.error("Need non-negative seeds and at least one whole episode per split.")
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    policy, scales, checkpoint, old_split, checkpoint_hash = load_policy(checkpoint_path, root)
    seeds = list(range(args.seed, args.seed + args.episodes))
    used_seeds = {r["seed"] for group in ("train", "validation") for r in old_split[group]}
    if used_seeds.intersection(seeds):
        raise ValueError("Choose new collection seeds, separate from original train/validation episodes.")
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "datasets/reach_grasp" / f"dagger_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "reach_grasp_dagger_batch_v1", "label": args.label,
        "method": "targeted closed-gripper on-policy expert-label collection, before aggregation/retraining",
        "behavior_checkpoint": str(checkpoint_path), "behavior_checkpoint_sha256": checkpoint_hash,
        "behavior_epoch": checkpoint["epoch"], "planned_seeds": seeds, "planned_episodes": len(seeds),
        "original_split_path": checkpoint["config"]["split_path"],
        "original_split_sha256": checkpoint["config"]["split_sha256"],
        "original_train_seeds": checkpoint["config"]["train_seeds"],
        "original_validation_seeds": checkpoint["config"]["validation_seeds"],
        "source_sha256": checkpoint["config"]["task_source_sha256"],
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256((root / "rollout_reach_grasp_bc.py").read_bytes()).hexdigest(),
        "label_selection": "previous_gripper_open < 0.5; independent of rollout success",
        "expert_labels_executed": False, "network_updated": False,
        "split_seed": args.split_seed, "validation_episodes": args.validation_episodes,
        "episodes": [], "total_rollout_states": 0, "total_labelled_states": 0,
        "behavior_policy_successes": 0, "completed_batch": False,
    }
    save_json(output / "manifest.json", manifest)
    print("Frozen learner executes actions; expert supplies labels only at closed-gripper states.", flush=True)
    for index, seed in enumerate(seeds):
        arrays, metadata = collect_episode(model, policy, scales, seed, checkpoint_hash)
        path = output / f"episode_seed{seed}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
        record = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "seed": seed, "labelled_states": metadata["labelled_states"],
                  "rollout_steps": metadata["rollout_steps"], "collection_complete": metadata["collection_complete"],
                  "behavior_policy_success": metadata["behavior_policy_success"], "weights_unchanged": unchanged}
        manifest["episodes"].append(record)
        manifest["total_rollout_states"] += metadata["rollout_steps"]
        manifest["total_labelled_states"] += metadata["labelled_states"]
        manifest["behavior_policy_successes"] += int(metadata["behavior_policy_success"])
        manifest["completed_batch"] = index + 1 == len(seeds)
        save_json(output / "manifest.json", manifest)
        print(f"[{index + 1:02d}/{len(seeds):02d}] seed={seed} labels={metadata['labelled_states']} "
              f"behavior_success={metadata['behavior_policy_success']} collection_complete={metadata['collection_complete']}", flush=True)
        if not unchanged:
            raise RuntimeError(f"Weights changed; partial records saved at {output}")
    manifest["checkpoint_file_unchanged"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    save_json(output / "manifest.json", manifest)
    if not manifest["checkpoint_file_unchanged"]:
        raise RuntimeError("Behavior checkpoint changed during collection.")
    if all(r["collection_complete"] and r["labelled_states"] > 0 for r in manifest["episodes"]):
        order = np.random.default_rng(args.split_seed).permutation(len(seeds))
        records = manifest["episodes"]
        split = {"schema": "reach_grasp_dagger_split_v1", "unit": "whole_episode", "split_seed": args.split_seed,
                 "train": [records[int(i)] for i in order[args.validation_episodes:]],
                 "validation": [records[int(i)] for i in order[:args.validation_episodes]]}
        save_json(output / "split.json", split)
        print(f"split = {output / 'split.json'}")
    else:
        print("Incomplete or unlabelled episodes retained; inspect before constructing a training split.")
    print("\n--- On-policy expert labels ---")
    print(f"rollout_states = {manifest['total_rollout_states']}")
    print(f"labelled_states = {manifest['total_labelled_states']}")
    print(f"behavior_policy_successes = {manifest['behavior_policy_successes']}/{len(seeds)} (collection diagnostic only)")
    print("weights_unchanged = True")
    print(f"manifest = {output / 'manifest.json'}")
    print("Failed learner rollouts retained. Expert labels were not executed. No policy retrained.")


if __name__ == "__main__":
    main()
