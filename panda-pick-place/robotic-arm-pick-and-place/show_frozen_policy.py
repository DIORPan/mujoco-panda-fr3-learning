"""Run the portable frozen model without requiring the original training data."""

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

from reach_grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, ReachGraspTask
from rollout_reach_grasp_bc import run_policy
from train_reach_grasp_bc import build_reach_grasp_policy


def load_frozen_model(path, project):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != "portable_panda_reach_grasp_policy_v1":
        raise ValueError("Expected a portable frozen reach-and-grasp policy.")
    if saved["observation_names"] != OBSERVATION_NAMES:
        raise ValueError("Observation schema mismatch.")
    scales = saved["observation_scales"].numpy()
    np.testing.assert_array_equal(scales, OBSERVATION_SCALES)
    for name, expected in saved["task_source_sha256"].items():
        source = (project / name).resolve()
        if not source.is_relative_to(project) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen execution source changed: {name}")
    if saved["mujoco"] != mujoco.__version__:
        raise ValueError("Use the recorded MuJoCo version.")
    if hashlib.sha256((project / "train_reach_grasp_bc.py").read_bytes()).hexdigest() != saved["model_builder_sha256"]:
        raise ValueError("Frozen network implementation changed.")
    torch.set_num_threads(1)
    policy = build_reach_grasp_policy()
    policy.load_state_dict(saved["model_state"])
    policy.eval().requires_grad_(False)
    return policy, scales, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bc", "aggregated"), default="aggregated")
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    project = Path(__file__).resolve().parent
    model_path = project.parent / "experiments/2026-10-24/checkpoints" / f"{args.model}.pt"
    policy, scales, metadata = load_frozen_model(model_path, project)
    os.chdir(project)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    task = ReachGraspTask(model)
    before = [p.detach().clone() for p in policy.parameters()]
    if args.viewer:
        import mujoco.viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, task.data)
    else:
        viewer_context = contextlib.nullcontext(None)
    with viewer_context as viewer:
        observation = task.reset(args.seed, viewer)
        trace, reason = run_policy(task, policy, scales, observation, viewer)
    result = task.result()
    unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    if not unchanged:
        raise RuntimeError("Frozen weights changed during inference.")
    success = bool(reason == "horizon_reached" and result["grasp_success"]
                   and np.isfinite(trace["final_observation"]).all())
    report = {"schema": "portable_panda_policy_rollout_v1", "model": args.model, "seed": args.seed,
              "stop_reason": reason, **result, "policy_success": success, "weights_unchanged": unchanged,
              "original_checkpoint_sha256": metadata["original_checkpoint_sha256"],
              "expert_actions_used": False, "role": "portable frozen-policy demonstration"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = project / "runs/bc" / f"frozen_demo_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"model = {args.model}; seed = {args.seed}; policy_success = {success}")
    print(f"min_lift_last_hold_m = {result['min_lift_last_hold_m']}")
    print(f"max_hold_displacement_m = {result['max_hold_displacement_m']}")
    print(f"weights_unchanged = {unchanged}")
    print(f"saved = {output}")


if __name__ == "__main__":
    main()
