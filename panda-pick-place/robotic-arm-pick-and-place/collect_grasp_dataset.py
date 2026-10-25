"""Collect feedback-expert demonstrations for the six-state/two-action grasp task."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from grasp_task import VerticalGraspTask, OBSERVATION_NAMES, OBSERVATION_SCALES, expert_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0 or args.split_seed < 0 or args.episodes < 2 or not 1 <= args.validation_episodes < args.episodes:
        parser.error("Need non-negative seeds and at least one whole episode per split.")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "datasets/grasp_only" / f"batch_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sources = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in (
        "grasp_task.py", "bc_observation.py", "controllers/scripted_controller.py",
        "assets/franka_emika_panda/scene.xml", "assets/franka_emika_panda/panda.xml",
        "collect_bc_episode.py", "collect_grasp_dataset.py")}
    records = []
    print("Collecting vertical grasp demos: 6 observations -> 2 actions, at 50 Hz.", flush=True)
    for index in range(args.episodes):
        seed = args.seed + index
        task = VerticalGraspTask(model)
        observation = task.reset(seed)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, task.data, initial, spec)
        initial_control = task.data.ctrl.copy()
        observations, actions = [], []
        for _ in range(int(round(task.HORIZON / task.CONTROL_DT))):
            observations.append(observation.copy())
            observation, applied = task.step(expert_action(observation))
            actions.append(applied)
        result = task.result()
        observations, actions = np.asarray(observations), np.asarray(actions)
        if not (np.isfinite(observations).all() and np.isfinite(actions).all()):
            raise RuntimeError("Non-finite recorded arrays.")
        metadata = {
            "schema": "vertical_grasp_episode_v1", "seed": seed,
            "task": "prepared-hover vertical grasp and hold",
            "setup": "scripted xy alignment and downward orientation, outside learned task",
            "observation_names": OBSERVATION_NAMES, "observation_scales": OBSERVATION_SCALES.tolist(),
            "action": "normalized z-reference increment [-1,1], gripper open command {0,1}",
            "max_delta_z_m": task.MAX_DELTA_Z, "control_dt_s": task.CONTROL_DT,
            "physics_dt_s": model.opt.timestep, "physics_substeps": task.substeps,
            "horizon_s": task.HORIZON, "fixed_xy_m": task.xy.tolist(),
            "initial_cube_z_m": task.initial_cube_z, "initial_z_reference_m": task.servo.HOVER_Z,
            "initial_state_spec": spec, "source_sha256": sources,
            "expert": "state-feedback expert_action, no phase or elapsed-time inputs",
            "mujoco": mujoco.__version__, "numpy": np.__version__, **result,
        }
        path = output / f"episode_seed{seed}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, observations=observations, actions=actions,
                                final_observation=observation, initial_state=initial,
                                initial_ctrl=initial_control, metadata_json=np.asarray(json.dumps(metadata)),
                                expert_success=np.asarray(result["grasp_success"]))
        record = {"file": path.name, "seed": seed, "samples": len(actions),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result}
        records.append(record)
        print(f"[{index + 1:02d}/{args.episodes:02d}] seed={seed} success={result['grasp_success']} "
              f"min_lift={result['min_lift_last_hold_m']:.3f} m", flush=True)
    manifest = {"schema": "vertical_grasp_batch_v1", "episodes": records,
                "successful_episodes": sum(record["grasp_success"] for record in records),
                "total_samples": sum(record["samples"] for record in records)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n--- Grasp dataset ---")
    print(f"expert_successes = {manifest['successful_episodes']}/{args.episodes}")
    print(f"observations: ({manifest['total_samples']}, 6); actions: ({manifest['total_samples']}, 2)")
    if manifest["successful_episodes"] == args.episodes:
        order = np.random.default_rng(args.split_seed).permutation(args.episodes)
        split = {"schema": "vertical_grasp_split_v1", "unit": "whole_episode", "split_seed": args.split_seed,
                 "train": [records[int(i)] for i in order[args.validation_episodes:]],
                 "validation": [records[int(i)] for i in order[:args.validation_episodes]]}
        (output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        print(f"train episodes = {len(split['train'])}; validation episodes = {len(split['validation'])}")
        print(f"split = {output / 'split.json'}")
    else:
        print("Some expert attempts failed. All retained; inspect before preparing training split.")
    print(f"manifest = {output / 'manifest.json'}")
    print("No neural policy trained.")


if __name__ == "__main__":
    main()
