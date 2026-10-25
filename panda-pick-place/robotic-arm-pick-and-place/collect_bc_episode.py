"""Record fixed or randomized green-cube expert episodes as NPZ files."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from bc_observation import ObservationReader
from controllers.scripted_controller import ScriptedController


# Small initial distribution for the first BC dataset, in world coordinates.
# Red/blue cubes and target pads stay fixed; green stays clear of those cubes.
GREEN_X_RANGE = (0.43, 0.55)
GREEN_Y_RANGE = (-0.08, 0.08)


def collect_episode(model, root, directory, seed, randomize_green, verbose_ik):
    data = mujoco.MjData(model)
    expert = ScriptedController(model, data, seed=seed)
    reader = ObservationReader(model)

    expert.reset(randomize=False)
    cube_joint = model.body("cube_green").jntadr[0]
    cube_address = int(model.jnt_qposadr[cube_joint])
    if randomize_green:
        rng = np.random.default_rng(seed)
        x = rng.uniform(*GREEN_X_RANGE)
        y = rng.uniform(*GREEN_Y_RANGE)
        data.qpos[cube_address:cube_address + 7] = [x, y, 0.25, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
    requested_cube_position = data.qpos[cube_address:cube_address + 3].copy()
    expert._settle(300)  # Scene settling is outside the recorded task episode.
    settled_cube_position = reader.fields(data)["cube_position (m)"]

    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial_state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, initial_state, state_spec)
    observations, actions, times = [], [], []

    def record_before_step(current):
        # The expert has assigned ctrl, but this ctrl has NOT yet been applied.
        # Observation contains state only; it does not contain the new ctrl.
        observation = reader.read(current)
        action = np.concatenate([current.ctrl[:7].copy(), [current.ctrl[7] / 255.0]])
        if not (np.isfinite(observation).all() and np.isfinite(action).all()):
            raise RuntimeError("Non-finite observation or action; episode not saved.")
        observations.append(observation)
        actions.append(action)
        times.append(current.time)

    expert.before_step = record_before_step
    try:
        # Keep the original IK diagnostics available without flooding batch logs.
        logging_context = contextlib.nullcontext() if verbose_ik else contextlib.redirect_stdout(io.StringIO())
        with logging_context:
            expert_success = expert.pick_and_place("cube_green", "target_green")
    finally:
        expert.before_step = None

    observations = np.stack(observations)
    actions = np.stack(actions)
    times = np.asarray(times)
    final_observation = reader.read(data)
    final_fields = reader.fields(data)
    cube = final_fields["cube_position (m)"]
    target = final_fields["target_position (m)"]
    xy_error = float(np.linalg.norm(cube[:2] - target[:2]))

    created = datetime.now(timezone.utc)
    source_paths = (
        "assets/franka_emika_panda/scene.xml",
        "assets/franka_emika_panda/panda.xml",
        "controllers/scripted_controller.py",
        "bc_observation.py",
        "collect_bc_episode.py",
    )
    metadata = {
        "schema": "panda_green_bc_v1",
        "created_utc": created.isoformat(),
        "expert": "ScriptedController, adapted from RafayDon upstream",
        "task": ["cube_green", "target_green"],
        "randomized": randomize_green,
        "seed": seed,
        "randomization": {
            "green_x_range_m": list(GREEN_X_RANGE) if randomize_green else None,
            "green_y_range_m": list(GREEN_Y_RANGE) if randomize_green else None,
            "other_cubes_and_targets": "fixed",
        },
        "requested_cube_position_m": requested_cube_position.tolist(),
        "settled_cube_position_m": settled_cube_position.tolist(),
        "observation_fields": list(final_fields),
        "observation_widths": [len(value) for value in final_fields.values()],
        "action": "7 absolute joint position targets (rad), 1 gripper command (0=close, 1=open)",
        "pairing": "observations[t] precedes actions[t] application; one action per physics step",
        "timestep_s": model.opt.timestep,
        "initial_settle_steps": 300,
        "initial_state_spec": state_spec,
        "success_criterion": "upstream: final cube xy error < 0.05 m and center z > 0.18 m; no stability test",
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
    }

    output = directory / f"green_{created.strftime('%Y%m%dT%H%M%S_%fZ')}_seed{seed}.npz"
    # Exclusive creation prevents overwriting earlier user or validation runs.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            observations=observations,
            actions=actions,
            times=times,
            initial_state=initial_state,
            final_observation=final_observation,
            expert_success=np.asarray(expert_success),
            final_xy_error_m=np.asarray(xy_error),
            final_cube_z_m=np.asarray(cube[2]),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )

    with np.load(output, allow_pickle=False) as saved:
        if saved["observations"].shape != (len(times), 25) or saved["actions"].shape != (len(times), 8):
            raise RuntimeError("Saved array dimensions do not match the expected schema.")
        if not np.allclose(np.diff(saved["times"]), model.opt.timestep, rtol=0, atol=1e-10):
            raise RuntimeError("Saved sample times are not spaced one physics step apart.")

    return {
        "file": output.name,
        "seed": seed,
        "requested_cube_position_m": requested_cube_position.tolist(),
        "settled_cube_position_m": settled_cube_position.tolist(),
        "samples": len(times),
        "expert_success": bool(expert_success),
        "final_xy_error_m": xy_error,
        "final_cube_z_m": float(cube[2]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="First seed; incremented by one per episode")
    parser.add_argument("--randomize-green", action="store_true", help="Vary only the green cube's starting xy")
    parser.add_argument("--verbose-ik", action="store_true", help="Show the existing per-iteration IK logs")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("--episodes must be positive and --seed must be non-negative.")

    # Keep MuJoCo's file paths relative on Windows (Chinese workspace path).
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    created = datetime.now(timezone.utc)
    directory = root / "datasets/bc" / f"batch_{created.strftime('%Y%m%dT%H%M%S_%fZ')}"
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Recording {args.episodes} expert episode(s); randomized green = {args.randomize_green}.", flush=True)
    results = []
    for index in range(args.episodes):
        result = collect_episode(model, root, directory, args.seed + index, args.randomize_green, args.verbose_ik)
        results.append(result)  # Preserve successful and failed attempts alike.
        x, y, _ = result["requested_cube_position_m"]
        status = "SUCCESS" if result["expert_success"] else "FAILED"
        print(
            f"[{index + 1:02d}/{args.episodes:02d}] seed={result['seed']} "
            f"start_xy=({x:.4f}, {y:.4f}) {status} "
            f"xy_error={result['final_xy_error_m']:.6f} m",
            flush=True,
        )

    manifest = {
        "schema": "panda_green_bc_batch_v1",
        "created_utc": created.isoformat(),
        "randomized_green": args.randomize_green,
        "first_seed": args.seed,
        "requested_episodes": args.episodes,
        "successful_episodes": sum(result["expert_success"] for result in results),
        "total_samples": sum(result["samples"] for result in results),
        "episodes": results,
    }
    manifest_path = directory / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    if args.episodes == 1:
        result = results[0]
        print("\n--- Saved episode ---")
        print(f"observations.shape = ({result['samples']}, 25)")
        print(f"actions.shape = ({result['samples']}, 8)")
        print(f"sample dt = {model.opt.timestep:.4f} s")
        print(f"recorded duration = {result['samples'] * model.opt.timestep:.3f} s (simulation time)")
        print(f"expert_success = {result['expert_success']} (original criterion)")
        print(f"saved = {directory / result['file']}")
    print("\n--- Saved batch ---")
    print(f"expert_successes = {manifest['successful_episodes']}/{args.episodes} (original criterion)")
    print(f"total_samples = {manifest['total_samples']}")
    print(f"manifest = {manifest_path}")
    print("All attempted episodes retained. No policy trained.")


if __name__ == "__main__":
    main()
