"""Record one feedback-expert reach-and-grasp demonstration; no neural policy."""

import argparse
import contextlib
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask, expert_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    task = ReachGraspTask(model)
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, task.data)
    else:
        viewer_context = contextlib.nullcontext(None)
    observations, actions, cube_positions, grasp_positions, references = [], [], [], [], []
    reason = "horizon_reached"
    with viewer_context as viewer:
        print("SCRIPTED SETUP: same fixed hover reference [0.40, 0, 0.40] for every cube.", flush=True)
        observation = task.reset(args.seed, viewer)
        print(f"Initial horizontal error = {task.initial_xy_error * 100:.2f} cm", flush=True)
        print("EXPERT TASK: horizontal approach, descend, close, lift and hold; no neural policy.", flush=True)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial_state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, task.data, initial_state, spec)
        initial_ctrl = task.data.ctrl.copy()
        wall_start = time.perf_counter()
        for index in range(int(round(task.HORIZON / task.CONTROL_DT))):
            if viewer is not None and not viewer.is_running():
                reason = "viewer_closed"
                break
            fields = task.reader.fields(task.data)
            reference = task.reference_position.copy()
            try:
                next_observation, applied = task.step(expert_action(observation), viewer)
            except InterruptedError:
                reason = "viewer_closed"
                break
            observations.append(observation.copy())
            actions.append(applied)
            cube_positions.append(fields["cube_position (m)"])
            grasp_positions.append(fields["grasp_position (m)"])
            references.append(reference)
            observation = next_observation
            if viewer is not None:
                delay = wall_start + (index + 1) * task.CONTROL_DT - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    result = task.result()
    observations = np.asarray(observations).reshape(-1, 10)
    actions = np.asarray(actions).reshape(-1, 4)
    if not (np.isfinite(observations).all() and np.isfinite(actions).all()):
        raise RuntimeError("Non-finite recorded expert data.")
    source_names = (
        "reach_grasp_task.py", "demo_reach_grasp.py", "grasp_task.py", "bc_observation.py",
        "collect_bc_episode.py", "controllers/scripted_controller.py",
        "assets/franka_emika_panda/scene.xml", "assets/franka_emika_panda/panda.xml",
    )
    report = {
        "schema": "reach_grasp_expert_demo_v1", "seed": args.seed,
        "controller": "feedback expert; no neural policy trained or executed",
        "task": "reach, descend, close, lift and hold from a fixed hover start",
        "setup_reference_m": task.START_POSITION.tolist(), "initial_cube_xy_m": task.requested_cube_xy.tolist(),
        "initial_grasp_position_m": task.initial_grasp_position.tolist(),
        "initial_xy_error_m": task.initial_xy_error,
        "observation_names": OBSERVATION_NAMES, "observation_scales": OBSERVATION_SCALES.tolist(),
        "actions": "three normalized reference increments [-1,1] and gripper open command {0,1}",
        "control_dt_s": task.CONTROL_DT, "physics_dt_s": model.opt.timestep,
        "physics_substeps": task.substeps, "max_delta_xyz_m": task.MAX_DELTA_XYZ,
        "horizon_s": task.HORIZON, "initial_state_spec": spec,
        "reference_bounds_m": [task.REFERENCE_LOWER.tolist(), task.REFERENCE_UPPER.tolist()],
        "stop_reason": reason, **result,
        "expert_success": bool(reason == "horizon_reached" and result["grasp_success"]),
        "horizontal_action_steps": int(np.count_nonzero(np.any(np.abs(actions[:, :2]) > 1e-10, axis=1))),
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "datasets/reach_grasp" / f"demo_{stamp}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "episode.npz", observations=observations, actions=actions,
                        cube_positions=np.asarray(cube_positions).reshape(-1, 3),
                        grasp_positions=np.asarray(grasp_positions).reshape(-1, 3),
                        reference_positions=np.asarray(references).reshape(-1, 3),
                        initial_state=initial_state, initial_ctrl=initial_ctrl,
                        final_observation=task.observe(), expert_success=np.asarray(report["expert_success"]),
                        metadata_json=np.asarray(json.dumps(report)))
    print("\n--- Reach-and-grasp expert demo ---")
    for key in ("expert_success", "stop_reason", "initial_xy_error_m", "horizontal_action_steps",
                "decisions", "min_lift_last_hold_m", "hold_duration_s", "max_hold_displacement_m"):
        print(f"{key} = {report[key]}")
    print(f"observations = {observations.shape}; actions = {actions.shape}")
    print(f"saved = {output}")
    print("Expert demonstration only. No 10-input/4-output neural policy trained or executed.")


if __name__ == "__main__":
    main()
