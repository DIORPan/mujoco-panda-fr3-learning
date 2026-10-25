"""Run the aggregated-label policy with the unchanged XYZ/gripper executor."""

import argparse
import contextlib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch

from collect_reach_grasp_dataset import json_safe
from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask
from rollout_reach_grasp_bc import run_policy
from train_reach_grasp_bc import build_reach_grasp_policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_dagger_policy(checkpoint_path, root):
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "reach_grasp_dagger_policy_v1":
        raise ValueError("Expected the aggregated-label 10-input/4-output policy.")
    config = checkpoint["config"]
    if config["observation_names"] != OBSERVATION_NAMES or mujoco.__version__ != config["mujoco"]:
        raise ValueError("Observation schema or MuJoCo version differs from training.")
    scales = checkpoint["observation_scales"].numpy()
    np.testing.assert_array_equal(scales, OBSERVATION_SCALES)
    for name, expected in config["task_source_sha256"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or digest(source) != expected:
            raise ValueError(f"Task source changed since training: {name}")
    if digest(root / "train_reach_grasp_bc.py") != config["model_builder_sha256"]:
        raise ValueError("Network implementation changed since training.")
    if digest(root / "train_reach_grasp_dagger.py") != config["trainer_sha256"]:
        raise ValueError("Aggregation trainer changed since training.")
    for source in config["data_sources"].values():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Training data provenance changed.")
    split_path = Path(config["split_path"])
    if digest(split_path) != config["split_sha256"]:
        raise ValueError("Aggregate split changed since training.")
    split = read_json(split_path)
    if split.get("schema") != "reach_grasp_aggregate_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected the aggregate whole-episode split.")
    records = split["train"] + split["validation"]
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Overlapping aggregate episode identities.")
    for group, key in (("train", "train_seeds"), ("validation", "validation_seeds")):
        if [record["seed"] for record in split[group]] != config[key]:
            raise ValueError("Split identities differ from the saved policy.")
    for record in records:
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Training episode changed: seed {record['seed']}")
    if (config["control_dt_s"] != ReachGraspTask.CONTROL_DT or config["max_delta_xyz_m"] != ReachGraspTask.MAX_DELTA_XYZ
            or config["horizon_s"] != ReachGraspTask.HORIZON):
        raise ValueError("Execution contract changed since training.")
    torch.set_num_threads(1)
    policy = build_reach_grasp_policy().to("cpu")
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval().requires_grad_(False)
    return policy, scales, checkpoint, split, checkpoint_hash


def find_baseline(root, config, seed):
    """Find a saved evaluation of the exact parent checkpoint and this seed."""
    plans = sorted((root / "runs/bc").glob("reach_grasp_eval_*/plan.json"), reverse=True)
    for path in plans:
        plan = read_json(path)
        if (plan.get("schema") == "reach_grasp_evaluation_v1"
                and plan.get("checkpoint_sha256") == config["initial_checkpoint_sha256"]
                and seed in plan.get("seeds", [])):
            summary = read_json(path.parent / "summary.json")
            matches = [item for item in summary["episodes"] if item["seed"] == seed]
            if len(matches) == 1:
                return path.parent, matches[0]
    return None, None


def post_lift_xy_displacement(trace):
    lifts = np.r_[trace["observations"][:, 3], trace["final_observation"][3]]
    cubes = np.vstack([trace["cube_positions"], trace["final_cube_position"]])
    reached = np.flatnonzero(lifts >= .10)
    if len(reached) == 0:
        return None
    start = int(reached[0])
    return float(np.linalg.norm(cubes[start:, :2] - cubes[start, :2], axis=1).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-seed", type=int, default=2002)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.episode_seed < 0:
        parser.error("--episode-seed must be non-negative.")
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    policy, scales, checkpoint, split, checkpoint_hash = load_dagger_policy(checkpoint_path, root)
    config = checkpoint["config"]
    group = next((name for name in ("train", "validation")
                  if any(r["seed"] == args.episode_seed for r in split[name])), "outside_training_and_validation")
    baseline_folder, baseline = find_baseline(root, config, args.episode_seed)
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    if model.opt.timestep != config["physics_dt_s"]:
        raise ValueError("Physics timestep differs from training.")
    task = ReachGraspTask(model)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, task.data)
    else:
        viewer_context = contextlib.nullcontext(None)
    print(f"Frozen aggregated-label policy; epoch={checkpoint['epoch']}; seed={args.episode_seed}; {group}", flush=True)
    with viewer_context as viewer:
        print("SCRIPTED SETUP: unchanged fixed hover [0.40, 0, 0.40].", flush=True)
        observation = task.reset(args.episode_seed, viewer)
        if baseline is not None:
            with np.load(baseline_folder / baseline["trace_file"], allow_pickle=False) as saved:
                np.testing.assert_allclose(observation, saved["observations"][0], rtol=0, atol=1e-10)
        print("POLICY CONTROL: unchanged XYZ/gripper action decoder and IK; no expert actions.", flush=True)
        trace, reason = run_policy(task, policy, scales, observation, viewer)
    result = task.result()
    finite = bool(np.isfinite(trace["final_observation"]).all()
                  and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    if not unchanged or digest(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Weights or checkpoint changed during evaluation.")
    lifts = np.r_[trace["observations"][:, 3], trace["final_observation"][3]]
    report = {
        "schema": "reach_grasp_dagger_rollout_v1", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "epoch": checkpoint["epoch"], "episode_seed": args.episode_seed,
        "dataset_group": group, "role": "diagnostic regression check; not an independent final test",
        "initial_observation_matches_parent_evaluation": baseline is not None,
        "baseline_evaluation": str(baseline_folder) if baseline_folder else None,
        "baseline_result": baseline, "setup_reference_m": task.START_POSITION.tolist(),
        "initial_xy_error_m": task.initial_xy_error, "stop_reason": reason, "finite_final_state": finite,
        "expert_actions_used_during_task": False, "weights_unchanged": unchanged, **result,
        "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
        "maximum_cube_lift_m": float(lifts.max()), "post_lift_cube_xy_displacement_m": post_lift_xy_displacement(trace),
        "gripper_command_switches": int(np.count_nonzero(np.diff(np.r_[1., trace["applied_actions"][:, 3]]))),
        "control_dt_s": task.CONTROL_DT, "max_delta_xyz_m": task.MAX_DELTA_XYZ,
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "viewer": args.viewer, "runner_sha256": digest(Path(__file__)),
        "executor_sha256": digest(root / "rollout_reach_grasp_bc.py"),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_dagger_rollout_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "trace.npz", **trace)
    print("\n--- Aggregated-label reach-and-grasp rollout ---")
    for key in ("stop_reason", "policy_success", "decisions", "initial_xy_error_m", "maximum_cube_lift_m",
                "min_lift_last_hold_m", "hold_duration_s", "max_hold_displacement_m",
                "post_lift_cube_xy_displacement_m", "gripper_command_switches", "weights_unchanged"):
        print(f"{key} = {report[key]}")
    if baseline is not None:
        print(f"parent_BC_success_on_same_seed = {baseline['policy_success']}")
        print(f"parent_BC_min_lift_last_hold_m = {baseline['min_lift_last_hold_m']}")
        print(f"parent_BC_max_hold_displacement_m = {baseline['max_hold_displacement_m']}")
    print(f"saved = {output}")
    print("One diagnostic regression episode. Fresh-seed evaluation still required.")


if __name__ == "__main__":
    main()
