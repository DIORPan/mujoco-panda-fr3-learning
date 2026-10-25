"""Collect whole-episode reach-and-grasp demonstrations; no neural training."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from collect_bc_episode import GREEN_X_RANGE, GREEN_Y_RANGE
from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask, expert_action


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def collect_episode(model, seed, sources):
    task = ReachGraspTask(model)
    observations, actions, cube_positions, grasp_positions, references = [], [], [], [], []
    initial_state, initial_ctrl = np.empty(0), np.empty(0)
    observation = np.full(10, np.nan)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    reason = "horizon_reached"
    try:
        observation = task.reset(seed)
        initial_state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, task.data, initial_state, spec)
        initial_ctrl = task.data.ctrl.copy()
        for _ in range(int(round(task.HORIZON / task.CONTROL_DT))):
            fields = task.reader.fields(task.data)
            reference = task.reference_position.copy()
            next_observation, applied = task.step(expert_action(observation))
            observations.append(observation.copy())
            actions.append(applied)
            cube_positions.append(fields["cube_position (m)"])
            grasp_positions.append(fields["grasp_position (m)"])
            references.append(reference)
            observation = next_observation
    except (RuntimeError, ValueError, FloatingPointError) as error:
        reason = f"execution_error: {error}"
    result = task.result() if task.pose_history else {
        "completed": False, "grasp_success": False, "decisions": getattr(task, "decisions", 0),
    }
    observations = np.asarray(observations).reshape(-1, 10)
    actions = np.asarray(actions).reshape(-1, 4)
    finite = bool(np.isfinite(observations).all() and np.isfinite(actions).all()
                  and np.isfinite(observation).all()
                  and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    bounded = bool(np.all(np.abs(actions[:, :3]) <= 1 + 1e-10)
                   and np.isin(actions[:, 3], [0., 1.]).all())
    success = bool(reason == "horizon_reached" and finite and bounded and result["grasp_success"])
    metadata = {
        "schema": "reach_grasp_episode_v1", "seed": seed,
        "task": "horizontal approach, descend, close, lift and hold from a fixed hover start",
        "setup": "scripted constant hover reference, independent of cube position",
        "setup_reference_m": task.START_POSITION.tolist(),
        "initial_cube_xy_m": task.requested_cube_xy.tolist() if hasattr(task, "requested_cube_xy") else None,
        "initial_cube_z_m": getattr(task, "initial_cube_z", None),
        "initial_xy_error_m": getattr(task, "initial_xy_error", None),
        "observation_names": OBSERVATION_NAMES, "observation_scales": OBSERVATION_SCALES.tolist(),
        "action": "three normalized reference increments [-1,1] and gripper open command {0,1}",
        "max_delta_xyz_m": task.MAX_DELTA_XYZ, "control_dt_s": task.CONTROL_DT,
        "physics_dt_s": model.opt.timestep, "physics_substeps": task.substeps,
        "horizon_s": task.HORIZON, "initial_state_spec": spec,
        "reference_bounds_m": [task.REFERENCE_LOWER.tolist(), task.REFERENCE_UPPER.tolist()],
        "expert": expert_action.__name__, "expert_inputs": "only recorded state features; no phase or elapsed time",
        "source_sha256": sources, "stop_reason": reason,
        "finite_recorded_and_final_state": finite, "bounded_actions": bounded,
        **result, "expert_success": success,
        "horizontal_action_steps": int(np.count_nonzero(np.any(np.abs(actions[:, :2]) > 1e-10, axis=1))),
        "mujoco": mujoco.__version__, "numpy": np.__version__,
    }
    arrays = {
        "observations": observations, "actions": actions,
        "cube_positions": np.asarray(cube_positions).reshape(-1, 3),
        "grasp_positions": np.asarray(grasp_positions).reshape(-1, 3),
        "reference_positions": np.asarray(references).reshape(-1, 3),
        "initial_state": initial_state, "initial_ctrl": initial_ctrl,
        "final_observation": observation, "expert_success": np.asarray(success),
        "metadata_json": np.asarray(json.dumps(json_safe(metadata), allow_nan=False)),
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--label", default="reach_grasp_expert_batch")
    args = parser.parse_args()
    if args.seed < 0 or args.split_seed < 0 or args.episodes < 2 or not 1 <= args.validation_episodes < args.episodes:
        parser.error("Need non-negative seeds and at least one whole episode per split.")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "datasets/reach_grasp" / f"batch_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    names = (
        "reach_grasp_task.py", "grasp_task.py", "bc_observation.py", "collect_bc_episode.py",
        "controllers/scripted_controller.py", "assets/franka_emika_panda/scene.xml",
        "assets/franka_emika_panda/panda.xml", "collect_reach_grasp_dataset.py",
    )
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    manifest = {
        "schema": "reach_grasp_batch_v1", "label": args.label,
        "planned_seeds": list(range(args.seed, args.seed + args.episodes)),
        "planned_episodes": args.episodes, "validation_episodes": args.validation_episodes,
        "split_seed": args.split_seed, "source_sha256": sources,
        "green_x_range_m": list(GREEN_X_RANGE), "green_y_range_m": list(GREEN_Y_RANGE),
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "episodes": [], "successful_episodes": 0, "total_samples": 0, "completed_batch": False,
    }
    save_json(output / "manifest.json", manifest)
    print("Collecting reach-and-grasp expert demos: 10 observations -> 4 actions, at 50 Hz.", flush=True)
    for index, seed in enumerate(manifest["planned_seeds"]):
        arrays, metadata = collect_episode(model, seed, sources)
        path = output / f"episode_seed{seed}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record = {"file": path.name, "seed": seed, "samples": len(arrays["actions"]),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  **{key: metadata[key] for key in (
                      "completed", "grasp_success", "expert_success", "stop_reason", "initial_xy_error_m",
                      "horizontal_action_steps", "decisions", "finite_recorded_and_final_state", "bounded_actions")}}
        for key in ("min_lift_last_hold_m", "hold_duration_s", "max_hold_displacement_m", "max_grasp_distance_last_hold_m"):
            record[key] = metadata.get(key)
        manifest["episodes"].append(record)
        manifest["successful_episodes"] += int(metadata["expert_success"])
        manifest["total_samples"] += len(arrays["actions"])
        manifest["completed_batch"] = index + 1 == args.episodes
        save_json(output / "manifest.json", manifest)
        print(f"[{index + 1:02d}/{args.episodes:02d}] seed={seed} success={metadata['expert_success']} "
              f"samples={record['samples']}; {metadata['stop_reason']}", flush=True)
    print("\n--- Reach-and-grasp dataset ---")
    print(f"expert_successes = {manifest['successful_episodes']}/{args.episodes}")
    print(f"observations: ({manifest['total_samples']}, 10); actions: ({manifest['total_samples']}, 4)")
    if manifest["successful_episodes"] == args.episodes:
        order = np.random.default_rng(args.split_seed).permutation(args.episodes)
        records = manifest["episodes"]
        split = {"schema": "reach_grasp_split_v1", "unit": "whole_episode", "split_seed": args.split_seed,
                 "train": [records[int(i)] for i in order[args.validation_episodes:]],
                 "validation": [records[int(i)] for i in order[:args.validation_episodes]]}
        save_json(output / "split.json", split)
        print(f"train episodes = {len(split['train'])}; validation episodes = {len(split['validation'])}")
        print(f"split = {output / 'split.json'}")
    else:
        print("Some expert attempts failed. All retained; inspect before preparing a training split.")
    print(f"manifest = {output / 'manifest.json'}")
    print("All attempted episodes retained. No 10-input/4-output neural policy trained.")


if __name__ == "__main__":
    main()
