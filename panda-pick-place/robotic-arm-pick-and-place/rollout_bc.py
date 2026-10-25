"""Run a frozen BC policy from one saved episode's exact initial state."""

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
import torch

from bc_observation import ObservationReader
from train_bc import build_policy


def training_target_rate_limits(model, folder, records, scale):
    """Derive per-joint command-rate limits from training episodes only."""
    peaks = np.zeros(7)
    for record in records:
        path = folder / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Training episode changed: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            initial = mujoco.MjData(model)
            mujoco.mj_setState(model, initial, saved["initial_state"], metadata["initial_state_spec"])
            commands = np.vstack([initial.ctrl[:7], saved["actions"][:, :7]])
            rates = np.abs(np.diff(commands, axis=0) / metadata["timestep_s"])
            peaks = np.maximum(peaks, rates.max(axis=0))
    return peaks * scale


@torch.inference_mode()
def run_policy(model, data, policy, reader, normalizers, steps, viewer, target_rates=None):
    obs_mean, obs_std, arm_mean, arm_std = normalizers
    joint_ids = [model.joint(f"joint{i}").id for i in range(1, 8)]
    limits = model.jnt_range[joint_ids]
    observations, actions, probabilities, raw_arm_targets = [], [], [], []
    clipped_steps, rate_limited_steps = 0, 0
    previous_target = data.ctrl[:7].copy()
    peak_applied_rates = np.zeros(7)
    reason = "horizon_reached"
    initial_time, wall_start = data.time, time.perf_counter()
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            reason = "viewer_closed"
            break
        obs = reader.read(data)
        normalized = ((obs - obs_mean) / obs_std).astype(np.float32)
        outputs = policy(torch.from_numpy(normalized).unsqueeze(0))[0]
        if not torch.isfinite(outputs).all().item():
            reason = "non_finite_prediction"
            break
        arm = outputs[:7].numpy() * arm_std + arm_mean
        bounded_arm = np.clip(arm, limits[:, 0], limits[:, 1])
        clipped_steps += int(np.any(arm != bounded_arm))
        if target_rates is not None:
            max_change = target_rates * model.opt.timestep
            limited_arm = previous_target + np.clip(bounded_arm - previous_target, -max_change, max_change)
            rate_limited_steps += int(np.any(np.abs(limited_arm - bounded_arm) > 1e-12))
            bounded_arm = limited_arm
        peak_applied_rates = np.maximum(peak_applied_rates, np.abs(bounded_arm - previous_target) / model.opt.timestep)
        previous_target = bounded_arm.copy()
        probability = outputs[7].sigmoid().item()
        gripper = float(probability >= 0.5)
        action = np.concatenate([bounded_arm, [gripper]])
        observations.append(obs)
        actions.append(action)
        probabilities.append(probability)
        raw_arm_targets.append(arm.copy())
        data.ctrl[:7] = action[:7]
        data.ctrl[7] = 255.0 * gripper
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            reason = "non_finite_physics"
            break
        if abs(data.time - (initial_time + (step + 1) * model.opt.timestep)) > 1e-8:
            reason = "unexpected_simulation_time_reset"
            break
        if viewer is not None and ((step + 1) % 10 == 0 or step + 1 == steps):
            viewer.sync()
            delay = wall_start + (step + 1) * model.opt.timestep - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    return (
        np.asarray(observations).reshape(-1, 25),
        np.asarray(actions).reshape(-1, 8),
        np.asarray(probabilities), clipped_steps, reason,
        np.asarray(raw_arm_targets).reshape(-1, 7), rate_limited_steps, peak_applied_rates,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episode-seed", type=int, default=104)
    parser.add_argument("--viewer", action="store_true", help="Show a paced passive viewer")
    parser.add_argument("--target-rate-scale", type=float,
                        help="Limit target changes to this multiple of each joint's training-data peak; default is no rate limit")
    args = parser.parse_args()
    if args.target_rate_scale is not None and (not np.isfinite(args.target_rate_scale) or args.target_rate_scale <= 0):
        parser.error("--target-rate-scale must be finite and positive.")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != "panda_green_bc_policy_v1":
        raise ValueError("Expected a v1 green-cube BC policy.")
    split_path = Path(checkpoint["config"]["split_path"])
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != checkpoint["config"]["split_sha256"]:
        raise ValueError("Dataset split changed since training.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    matches = [
        (group, record)
        for group in ("train", "validation")
        for record in split[group] if record["seed"] == args.episode_seed
    ]
    if len(matches) != 1:
        raise ValueError("Choose exactly one episode seed from the saved split.")
    group, record = matches[0]
    episode_path = split_path.parent / record["file"]
    if hashlib.sha256(episode_path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Reference episode changed since splitting.")
    with np.load(episode_path, allow_pickle=False) as episode:
        metadata = json.loads(str(episode["metadata_json"]))
        initial_state = episode["initial_state"].copy()
        initial_observation = episode["observations"][0].copy()
        steps = len(episode["times"])

    root = Path(__file__).resolve().parent
    os.chdir(root)  # MuJoCo paths stay relative on Windows.
    for path in ("assets/franka_emika_panda/scene.xml", "assets/franka_emika_panda/panda.xml", "bc_observation.py"):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != metadata["source_sha256"][path]:
            raise ValueError(f"Model or observation source changed since collection: {path}")
    if mujoco.__version__ != metadata["mujoco"]:
        raise ValueError("Use the same MuJoCo version as the reference episode.")
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    if model.nu != 8 or model.opt.timestep != metadata["timestep_s"]:
        raise ValueError("Unexpected actuator count or physics timestep.")
    data = mujoco.MjData(model)
    spec = metadata["initial_state_spec"]
    if len(initial_state) != mujoco.mj_stateSize(model, spec):
        raise ValueError("Initial state does not match this model.")
    mujoco.mj_setState(model, data, initial_state, spec)
    mujoco.mj_forward(model, data)
    # Populate derived quantities for rendering, then restore integration state.
    mujoco.mj_setState(model, data, initial_state, spec)
    reader = ObservationReader(model)
    np.testing.assert_allclose(reader.read(data), initial_observation, rtol=0, atol=1e-10)
    target_rates = None
    if args.target_rate_scale is not None:
        target_rates = training_target_rate_limits(model, split_path.parent, split["train"], args.target_rate_scale)

    torch.set_num_threads(1)
    policy = build_policy().to("cpu")
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    before_weights = [parameter.detach().clone() for parameter in policy.parameters()]
    normalizers = tuple(checkpoint[name].numpy() for name in ("obs_mean", "obs_std", "arm_mean", "arm_std"))
    print(f"Frozen BC policy; epoch={checkpoint['epoch']}; {group} seed={args.episode_seed}", flush=True)
    print(f"Horizon = {steps * model.opt.timestep:.3f} simulation seconds; no expert actions used.", flush=True)
    if target_rates is not None:
        print(f"Per-joint target-rate limits (rad/s): {np.round(target_rates, 6).tolist()}", flush=True)
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)
    else:
        viewer_context = contextlib.nullcontext(None)
    with viewer_context as viewer:
        observations, actions, probabilities, clipped_steps, reason, raw_arm_targets, rate_limited_steps, peak_applied_rates = run_policy(
            model, data, policy, reader, normalizers, steps, viewer, target_rates,
        )
    final = reader.read(data)
    finite_final = bool(np.isfinite(final).all())
    xy_error = float(np.linalg.norm(final[19:21] - final[22:24])) if finite_final else None
    cube_z = float(final[21]) if finite_final else None
    completed = reason == "horizon_reached" and len(actions) == steps
    criterion = bool(finite_final and xy_error < 0.05 and cube_z > 0.18)
    unchanged = all(torch.equal(before, after) for before, after in zip(before_weights, policy.parameters()))
    if not unchanged:
        raise RuntimeError("Policy parameters changed during evaluation.")
    max_height = float(max(observations[:, 21].max(), cube_z)) if len(observations) and finite_final else None
    minimum_distance = float(np.linalg.norm(observations[:, 16:19] - observations[:, 19:22], axis=1).min()) if len(observations) else None
    report = {
        "schema": "panda_green_bc_rollout_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "epoch": checkpoint["epoch"], "reference_episode": str(episode_path),
        "episode_seed": args.episode_seed, "dataset_group": group,
        "role": "diagnostic rollout; not an independent final test",
        "requested_steps": steps, "executed_steps": len(actions),
        "timestep_s": model.opt.timestep, "stop_reason": reason,
        "completed_horizon": completed, "finite_final_state": finite_final,
        "final_xy_error_m": xy_error, "final_cube_z_m": cube_z,
        "maximum_cube_z_m": max_height, "minimum_grasp_cube_distance_m": minimum_distance,
        "original_criterion_at_stop": criterion,
        "policy_success": bool(completed and criterion),
        "success_criterion": "cube xy error < 0.05 m and center z > 0.18 m; no stability test",
        "clipped_target_steps": clipped_steps,
        "target_rate_scale": args.target_rate_scale,
        "target_rate_limits_rad_s": target_rates.tolist() if target_rates is not None else None,
        "target_rate_source": "training expert commands only" if target_rates is not None else None,
        "rate_limited_steps": rate_limited_steps,
        "peak_applied_target_rates_rad_s": peak_applied_rates.tolist(),
        "max_applied_target_rate_rad_s": float(peak_applied_rates.max()),
        "max_actual_joint_speed_rad_s": float(np.abs(observations[:, 7:14]).max()) if len(observations) else None,
        "gripper_command_switches": int(np.count_nonzero(np.diff(actions[:, 7]))) if len(actions) else 0,
        "action_processing": "clip 7 targets to joint ranges, optionally limit changes relative to previous ctrl; gripper probability >= 0.5 means open",
        "expert_actions_used": False, "weights_unchanged": unchanged,
        "viewer": args.viewer, "torch": str(torch.__version__), "mujoco": mujoco.__version__, "numpy": np.__version__,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"rollout_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "trace.npz", observations=observations, actions=actions,
                        gripper_open_probabilities=probabilities, final_observation=final,
                        raw_arm_targets=raw_arm_targets)
    print("\n--- BC rollout ---")
    for key in ("stop_reason", "policy_success", "final_xy_error_m", "final_cube_z_m",
                "maximum_cube_z_m", "gripper_command_switches", "clipped_target_steps", "rate_limited_steps",
                "max_applied_target_rate_rad_s", "max_actual_joint_speed_rad_s", "weights_unchanged"):
        print(f"{key} = {report[key]}")
    print(f"saved = {output}")
    print("Single diagnostic rollout. Report failures as observed; no aggregate success rate.")


if __name__ == "__main__":
    main()
