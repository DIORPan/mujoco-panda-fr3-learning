"""Run one frozen reach-and-grasp BC policy from an exact recorded start state."""

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

from collect_reach_grasp_dataset import json_safe
from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask
from train_reach_grasp_bc import build_reach_grasp_policy, load_episodes


def load_policy(checkpoint_path, root):
    """Validate the frozen checkpoint, saved split and current execution sources."""
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "reach_grasp_policy_v1":
        raise ValueError("Expected the 10-input/4-output reach-and-grasp checkpoint.")
    config = checkpoint["config"]
    if config["observation_names"] != OBSERVATION_NAMES:
        raise ValueError("Policy observation schema differs from this task.")
    scales = checkpoint["observation_scales"].numpy()
    np.testing.assert_array_equal(scales, OBSERVATION_SCALES)
    for name, digest in config["task_source_sha256"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Task source changed since training: {name}")
    if mujoco.__version__ != config["mujoco"]:
        raise ValueError("Use the same MuJoCo version as collection/training.")
    split_path = Path(config["split_path"])
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != config["split_sha256"]:
        raise ValueError("Dataset split changed since training.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "reach_grasp_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected a reach-and-grasp whole-episode split.")
    for group, key in (("train", "train_seeds"), ("validation", "validation_seeds")):
        if [record["seed"] for record in split[group]] != config[key]:
            raise ValueError("Recorded split identities differ from training config.")
    torch.set_num_threads(1)
    policy = build_reach_grasp_policy().to("cpu")
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval().requires_grad_(False)
    return policy, scales, checkpoint, split, checkpoint_hash


@torch.inference_mode()
def run_policy(task, policy, scales, observation, viewer=None):
    observations, requested, applied = [], [], []
    cubes, grasps, references = [], [], []
    reason = "horizon_reached"
    wall_start = time.perf_counter()
    for index in range(int(round(task.HORIZON / task.CONTROL_DT))):
        if viewer is not None and not viewer.is_running():
            reason = "viewer_closed"
            break
        if not np.isfinite(observation).all():
            reason = "non_finite_observation"
            break
        inputs = torch.from_numpy((observation / scales).astype(np.float32))
        outputs = policy(inputs)
        if outputs.shape != (4,) or not torch.isfinite(outputs).all().item():
            reason = "invalid_policy_prediction"
            break
        action = np.concatenate([outputs[:3].tanh().numpy(), [outputs[3].sigmoid().item()]])
        fields = task.reader.fields(task.data)
        reference = task.reference_position.copy()
        try:
            next_observation, executed = task.step(action, viewer)
        except InterruptedError:
            reason = "viewer_closed"
            break
        except RuntimeError as error:
            reason = f"execution_error: {error}"
            break
        observations.append(observation.copy())
        requested.append(action)
        applied.append(executed)
        cubes.append(fields["cube_position (m)"])
        grasps.append(fields["grasp_position (m)"])
        references.append(reference)
        observation = next_observation
        if viewer is not None:
            delay = wall_start + (index + 1) * task.CONTROL_DT - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    fields = task.reader.fields(task.data)
    return {
        "observations": np.asarray(observations).reshape(-1, 10),
        "requested_actions": np.asarray(requested).reshape(-1, 4),
        "applied_actions": np.asarray(applied).reshape(-1, 4),
        "cube_positions": np.asarray(cubes).reshape(-1, 3),
        "grasp_positions": np.asarray(grasps).reshape(-1, 3),
        "reference_positions": np.asarray(references).reshape(-1, 3),
        "final_observation": task.observe(), "final_cube_position": fields["cube_position (m)"],
        "final_grasp_position": fields["grasp_position (m)"],
        "final_reference_position": task.reference_position.copy(),
    }, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-seed", type=int, default=311)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    policy, scales, checkpoint, split, checkpoint_hash = load_policy(checkpoint_path, root)
    split_path = Path(checkpoint["config"]["split_path"])
    matches = [(group, record) for group in ("train", "validation")
               for record in split[group] if record["seed"] == args.episode_seed]
    if len(matches) != 1:
        raise ValueError("Choose exactly one recorded seed from this checkpoint's split.")
    group, record = matches[0]
    reference_obs, _, metadata = load_episodes(split_path.parent, [record], root)
    if metadata["source_sha256"] != checkpoint["config"]["task_source_sha256"]:
        raise ValueError("Checkpoint and reference episode task versions differ.")
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    task = ReachGraspTask(model)
    if model.opt.timestep != metadata["physics_dt_s"] or task.substeps != metadata["physics_substeps"]:
        raise ValueError("Physics/control timing differs from collection.")
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, task.data)
    else:
        viewer_context = contextlib.nullcontext(None)
    print(f"Frozen reach-and-grasp BC; epoch={checkpoint['epoch']}; {group} seed={args.episode_seed}", flush=True)
    with viewer_context as viewer:
        print("SCRIPTED SETUP: fixed hover reference [0.40, 0, 0.40], independent of cube position.", flush=True)
        observation = task.reset(args.episode_seed, viewer)
        episode_path = split_path.parent / record["file"]
        with np.load(episode_path, allow_pickle=False) as episode:
            spec = metadata["initial_state_spec"]
            initial = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, task.data, initial, spec)
            np.testing.assert_allclose(initial, episode["initial_state"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(task.data.ctrl, episode["initial_ctrl"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(observation, reference_obs[0], rtol=0, atol=1e-10)
        print(f"Initial horizontal error = {task.initial_xy_error * 100:.2f} cm", flush=True)
        print("POLICY CONTROL: XYZ increments and gripper; no expert action fallback.", flush=True)
        trace, reason = run_policy(task, policy, scales, observation, viewer)
    result = task.result()
    finite = bool(np.isfinite(trace["final_observation"]).all()
                  and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    if not unchanged or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Policy weights or checkpoint changed during evaluation.")
    commands = np.concatenate([[1.], trace["applied_actions"][:, 3]])
    lifts = np.concatenate([trace["observations"][:, 3], [trace["final_observation"][3]]])
    xy_errors = np.linalg.norm(np.concatenate([
        trace["observations"][:, :2], trace["final_observation"][None, :2]], axis=0), axis=1)
    report = {
        "schema": "reach_grasp_rollout_v1", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "epoch": checkpoint["epoch"],
        "episode_seed": args.episode_seed, "dataset_group": group,
        "reference_episode": str(episode_path), "initial_state_verified": True,
        "role": "one recorded-seed diagnostic; not an independent final test",
        "setup": "same fixed hover reference for every cube; no scripted horizontal alignment",
        "control": "frozen network chooses XYZ increments and gripper; IK keeps downward orientation",
        "setup_reference_m": task.START_POSITION.tolist(), "initial_xy_error_m": task.initial_xy_error,
        "expert_actions_used_during_task": False, "weights_unchanged": unchanged,
        "stop_reason": reason, "finite_final_state": finite, **result,
        "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
        "maximum_cube_lift_m": float(lifts.max()), "minimum_xy_error_m": float(xy_errors.min()),
        "final_xy_error_m": float(xy_errors[-1]),
        "horizontal_action_steps": int(np.count_nonzero(np.any(np.abs(trace["applied_actions"][:, :2]) > 1e-10, axis=1))),
        "gripper_command_switches": int(np.count_nonzero(np.diff(commands))),
        "control_dt_s": task.CONTROL_DT, "max_delta_xyz_m": task.MAX_DELTA_XYZ,
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "viewer": args.viewer, "mujoco": mujoco.__version__, "torch": str(torch.__version__),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_rollout_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "trace.npz", **trace)
    print("\n--- Reach-and-grasp BC rollout ---")
    for key in ("stop_reason", "policy_success", "decisions", "initial_xy_error_m", "minimum_xy_error_m",
                "maximum_cube_lift_m", "min_lift_last_hold_m", "hold_duration_s", "max_hold_displacement_m",
                "max_grasp_distance_last_hold_m", "gripper_command_switches", "weights_unchanged"):
        print(f"{key} = {report[key]}")
    print(f"saved = {output}")
    print("One diagnostic episode only; no aggregate success rate.")


if __name__ == "__main__":
    main()
