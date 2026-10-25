"""Evaluate one frozen vertical-grasp policy from a recorded hover state."""

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

from grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, VerticalGraspTask
from train_grasp_bc import build_grasp_policy, load_episodes


@torch.inference_mode()
def run_policy(task, policy, scales, observation, viewer):
    observations, requested, applied = [], [], []
    cube_positions, grasp_positions = [], []
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
        if not torch.isfinite(outputs).all().item():
            reason = "non_finite_prediction"
            break
        action = np.asarray([outputs[0].tanh().item(), outputs[1].sigmoid().item()])
        fields = task.reader.fields(task.data)
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
        cube_positions.append(fields["cube_position (m)"])
        grasp_positions.append(fields["grasp_position (m)"])
        observation = next_observation
        if viewer is not None:
            delay = wall_start + (index + 1) * task.CONTROL_DT - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    return {
        "observations": np.asarray(observations).reshape(-1, 6),
        "requested_actions": np.asarray(requested).reshape(-1, 2),
        "applied_actions": np.asarray(applied).reshape(-1, 2),
        "cube_positions": np.asarray(cube_positions).reshape(-1, 3),
        "grasp_positions": np.asarray(grasp_positions).reshape(-1, 3),
        "final_observation": task.observe(),
    }, reason


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-seed", type=int, default=204)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "vertical_grasp_policy_v1":
        raise ValueError("Expected the new six-input/two-output vertical-grasp checkpoint.")
    config = checkpoint["config"]
    if config["observation_names"] != OBSERVATION_NAMES:
        raise ValueError("Policy observation schema differs from this task.")
    scales = checkpoint["observation_scales"].numpy()
    np.testing.assert_array_equal(scales, OBSERVATION_SCALES)
    split_path = Path(config["split_path"])
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != config["split_sha256"]:
        raise ValueError("Dataset split changed since training.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "vertical_grasp_split_v1" or split.get("unit") != "whole_episode":
        raise ValueError("Expected a vertical grasp whole-episode split.")
    matches = [(group, record) for group in ("train", "validation")
               for record in split[group] if record["seed"] == args.episode_seed]
    if len(matches) != 1:
        raise ValueError("Choose exactly one recorded seed from this policy's split.")
    group, record = matches[0]
    reference_obs, _, metadata = load_episodes(split_path.parent, [record], root)
    if metadata["source_sha256"] != config["task_source_sha256"]:
        raise ValueError("Policy and reference task sources differ.")
    if mujoco.__version__ != metadata["mujoco"]:
        raise ValueError("Use the same MuJoCo version as collection.")
    os.chdir(root)  # Relative XML paths support this Windows workspace.
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    task = VerticalGraspTask(model)
    if model.opt.timestep != metadata["physics_dt_s"] or task.substeps != metadata["physics_substeps"]:
        raise ValueError("Physics/control contract changed since collection.")
    torch.set_num_threads(1)
    policy = build_grasp_policy().to("cpu")
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, task.data)
    else:
        viewer_context = contextlib.nullcontext(None)
    print(f"Frozen grasp BC policy; epoch={checkpoint['epoch']}; {group} seed={args.episode_seed}", flush=True)
    with viewer_context as viewer:
        print("SCRIPTED SETUP: align above the cube; outside the learned task.", flush=True)
        observation = task.reset(args.episode_seed, viewer)
        episode_path = split_path.parent / record["file"]
        with np.load(episode_path, allow_pickle=False) as episode:
            spec = metadata["initial_state_spec"]
            state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, task.data, state, spec)
            np.testing.assert_allclose(state, episode["initial_state"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(task.data.ctrl, episode["initial_ctrl"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(observation, reference_obs[0], rtol=0, atol=1e-10)
        print("POLICY CONTROL: descend, close, lift and hold; no expert action fallback.", flush=True)
        trace, reason = run_policy(task, policy, scales, observation, viewer)
    result = task.result()
    finite = bool(np.isfinite(trace["final_observation"]).all()
                  and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    if not unchanged or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Policy weights or saved checkpoint changed during evaluation.")
    commands = np.concatenate([[1.0], trace["applied_actions"][:, 1]])
    lifts = np.concatenate([trace["observations"][:, 1], [trace["final_observation"][1]]])
    report = {
        "schema": "vertical_grasp_rollout_v1", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "epoch": checkpoint["epoch"],
        "episode_seed": args.episode_seed, "dataset_group": group,
        "reference_episode": str(episode_path), "initial_state_verified": True,
        "role": "single diagnostic rollout on recorded seed; not an independent final test",
        "setup": "scripted hover alignment, outside learned task",
        "control": "frozen policy chooses vertical increment and gripper; fixed xy/downward orientation and IK servo",
        "expert_actions_used_during_task": False, "weights_unchanged": unchanged,
        "stop_reason": reason, "finite_final_state": finite,
        **result,
        "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
        "maximum_cube_lift_m": float(lifts.max()),
        "gripper_command_switches": int(np.count_nonzero(np.diff(commands))),
        "control_dt_s": task.CONTROL_DT, "max_delta_z_m": task.MAX_DELTA_Z,
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, cube displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "viewer": args.viewer, "mujoco": mujoco.__version__, "torch": str(torch.__version__),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"grasp_rollout_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "trace.npz", **trace)
    print("\n--- Grasp BC rollout ---")
    for key in ("stop_reason", "policy_success", "decisions", "maximum_cube_lift_m",
                "min_lift_last_hold_m", "hold_duration_s", "max_hold_displacement_m",
                "max_grasp_distance_last_hold_m", "gripper_command_switches", "weights_unchanged"):
        print(f"{key} = {report[key]}")
    print(f"saved = {output}")
    print("One diagnostic episode only; no aggregate success rate.")


if __name__ == "__main__":
    main()
