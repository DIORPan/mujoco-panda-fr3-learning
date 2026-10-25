"""Demonstrate and record grasp-only task boundaries with the scripted expert."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from bc_observation import ObservationReader
from collect_bc_episode import GREEN_X_RANGE, GREEN_Y_RANGE
from controllers.scripted_controller import ScriptedController


def quiet_ik(expert, target, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        return expert.ik(target, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    expert = ScriptedController(model, data, seed=args.seed)
    reader = ObservationReader(model)
    expert.reset(randomize=False)
    rng = np.random.default_rng(args.seed)
    x, y = rng.uniform(*GREEN_X_RANGE), rng.uniform(*GREEN_Y_RANGE)
    cube_joint = model.body("cube_green").jntadr[0]
    address = int(model.jnt_qposadr[cube_joint])
    data.qpos[address:address + 7] = [x, y, 0.25, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)
    else:
        viewer_context = contextlib.nullcontext(None)

    observations, actions, times, phases = [], [], [], []
    hold_positions, hold_grasp_positions = [], []
    phase = "setup"
    with viewer_context as viewer:
        print("SCRIPTED SETUP: positioning above green cube, outside the grasp-only task.", flush=True)
        expert._settle(300, viewer)
        initial_cube_z = float(reader.fields(data)["cube_position (m)"][2])
        q_hover = quiet_ik(expert, [x, y, expert.HOVER_Z], expert.HOME)
        expert._move_to(q_hover, expert.GRIPPER_OPEN, 1.5, viewer)
        expert._settle(100, viewer)
        if viewer is not None and not viewer.is_running():
            raise SystemExit("Viewer closed during setup; no task episode saved.")

        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial_state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, initial_state, spec)
        initial_ctrl = data.ctrl.copy()
        task_start_time = data.time
        hold_steps = int(round(0.5 / model.opt.timestep))
        expected_steps = 2 * int(1.5 / model.opt.timestep) + int(1.0 / model.opt.timestep) + 400 + hold_steps

        def record_before_step(current):
            fields = reader.fields(current)
            observations.append(np.concatenate(list(fields.values())))
            actions.append(np.concatenate([current.ctrl[:7].copy(), [current.ctrl[7] / 255.0]]))
            times.append(current.time)
            phases.append(phase)  # Diagnostic labels, not a learned policy input.
            if phase == "hold":
                hold_positions.append(fields["cube_position (m)"])
                hold_grasp_positions.append(fields["grasp_position (m)"])

        expert.before_step = record_before_step
        try:
            phase = "descend"
            print("TASK 1: descend with gripper open.", flush=True)
            q_grasp = quiet_ik(expert, [x, y, expert.GRASP_Z], q_hover)
            expert._move_to(q_grasp, expert.GRIPPER_OPEN, 1.5, viewer)

            phase = "close"
            print("TASK 2: close and settle.", flush=True)
            expert._move_to(q_grasp, expert.GRIPPER_CLOSE, 1.0, viewer)
            expert._settle(200, viewer)

            phase = "lift"
            print("TASK 3: lift and settle.", flush=True)
            q_lift = quiet_ik(expert, [x, y, expert.HOVER_Z], q_grasp)
            expert._move_to(q_lift, expert.GRIPPER_CLOSE, 1.5, viewer)
            phase = "stabilize"
            expert._settle(200, viewer)

            phase = "hold"
            print("TASK 4: measure a 0.5-second hold.", flush=True)
            hold_start_time = data.time
            expert._settle(hold_steps, viewer)
        finally:
            expert.before_step = None

    final_fields = reader.fields(data)
    hold_positions.append(final_fields["cube_position (m)"])
    hold_grasp_positions.append(final_fields["grasp_position (m)"])
    hold_positions = np.asarray(hold_positions)
    hold_grasp_positions = np.asarray(hold_grasp_positions)
    min_lift = float(hold_positions[:, 2].min() - initial_cube_z)
    hold_drift = float(np.linalg.norm(hold_positions - hold_positions[0], axis=1).max())
    max_grasp_distance = float(np.linalg.norm(hold_positions - hold_grasp_positions, axis=1).max())
    hold_duration = float(data.time - hold_start_time)
    completed = len(actions) == expected_steps and len(hold_positions) == hold_steps + 1
    completed = bool(completed and hold_duration >= 0.5 - 1e-8)
    success = bool(completed and min_lift >= 0.10 and hold_drift <= 0.01 and max_grasp_distance <= 0.04)
    observation_array = np.asarray(observations).reshape(-1, 25)
    action_array = np.asarray(actions).reshape(-1, 8)
    if not (np.isfinite(observation_array).all() and np.isfinite(action_array).all()):
        raise RuntimeError("Non-finite expert data; inspect the simulation.")

    report = {
        "schema": "panda_green_grasp_demo_v1",
        "task": "descend, close, lift and hold green cube from a prepared hover pose",
        "seed": args.seed, "requested_start_xy_m": [x, y],
        "controller": "scripted expert; no neural policy used",
        "setup": "scripted hover positioning; excluded from recorded task",
        "observation_schema": "panda_green_bc_v1 raw 25 values; task-specific policy preprocessing is a later step",
        "initial_state_spec": spec, "timestep_s": model.opt.timestep,
        "recorded_steps": len(actions), "expected_steps": expected_steps,
        "recorded_duration_s": float(data.time - task_start_time),
        "completed": completed, "expert_grasp_success": success,
        "initial_cube_z_m": initial_cube_z, "min_lift_during_hold_m": min_lift,
        "hold_duration_s": hold_duration, "max_hold_displacement_m": hold_drift,
        "max_cube_grasp_distance_during_hold_m": max_grasp_distance,
        "criterion": "over a complete 0.5 s hold: lift >= 0.10 m, cube displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "phase_labels": "diagnostic only; do not treat expert phase as an observation available to a learned policy",
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in ("assets/franka_emika_panda/scene.xml", "assets/franka_emika_panda/panda.xml",
                         "controllers/scripted_controller.py", "bc_observation.py", "collect_bc_episode.py", "demo_grasp_only.py")
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "datasets/grasp_only" / f"demo_{stamp}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "episode.npz", observations=observation_array, actions=action_array,
        times=np.asarray(times), phases=np.asarray(phases, dtype="U16"),
        initial_state=initial_state, initial_ctrl=initial_ctrl,
        final_observation=np.concatenate(list(final_fields.values())),
        expert_success=np.asarray(success), metadata_json=np.asarray(json.dumps(report)),
    )
    print("\n--- Grasp-only expert demo ---")
    for key in ("completed", "expert_grasp_success", "recorded_steps", "min_lift_during_hold_m",
                "hold_duration_s", "max_hold_displacement_m", "max_cube_grasp_distance_during_hold_m"):
        print(f"{key} = {report[key]}")
    print(f"saved = {output}")
    print("Expert reference only. No BC policy trained or executed.")


if __name__ == "__main__":
    main()
