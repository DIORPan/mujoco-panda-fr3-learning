"""Evaluate a frozen vertical-grasp checkpoint on a declared batch of new seeds."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch

from collect_bc_episode import GREEN_X_RANGE, GREEN_Y_RANGE
from grasp_task import OBSERVATION_NAMES, OBSERVATION_SCALES, VerticalGraspTask
from rollout_grasp_bc import json_safe, run_policy
from train_grasp_bc import build_grasp_policy, load_episodes


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--label", default="unseen_seed_evaluation")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Need positive --episodes and non-negative --seed.")
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "vertical_grasp_policy_v1":
        raise ValueError("Expected the six-input/two-output vertical-grasp checkpoint.")
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
    used_seeds = {record["seed"] for group in ("train", "validation") for record in split[group]}
    seeds = list(range(args.seed, args.seed + args.episodes))
    if used_seeds.intersection(seeds):
        raise ValueError("Evaluation seeds overlap training or validation; choose new seeds.")
    # Validate the task contract against collection; reference actions are never executed.
    _, _, metadata = load_episodes(split_path.parent, [split["train"][0]], root)
    if metadata["source_sha256"] != config["task_source_sha256"]:
        raise ValueError("Policy and collection task sources differ.")
    if mujoco.__version__ != metadata["mujoco"]:
        raise ValueError("Use the same MuJoCo version as collection.")
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    if model.opt.timestep != metadata["physics_dt_s"]:
        raise ValueError("Physics timestep differs from collection.")
    torch.set_num_threads(1)
    policy = build_grasp_policy().to("cpu")
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval().requires_grad_(False)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"grasp_eval_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "vertical_grasp_evaluation_v1", "label": args.label,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "epoch": checkpoint["epoch"], "seeds": seeds, "planned_episodes": len(seeds),
        "training_seeds": config["train_seeds"], "validation_seeds": config["validation_seeds"],
        "seed_overlap": False, "green_x_range_m": list(GREEN_X_RANGE), "green_y_range_m": list(GREEN_Y_RANGE),
        "scope": "new seeds from collection's xy distribution; scripted hover alignment, learned vertical grasp",
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "control_dt_s": VerticalGraspTask.CONTROL_DT, "horizon_s": VerticalGraspTask.HORIZON,
        "max_delta_z_m": VerticalGraspTask.MAX_DELTA_Z, "physics_dt_s": model.opt.timestep,
        "expert_actions_used_during_task": False, "task_source_sha256": config["task_source_sha256"],
        "evaluation_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                     for name in ("evaluate_grasp_bc.py", "rollout_grasp_bc.py", "train_grasp_bc.py")},
        "torch": str(torch.__version__), "mujoco": mujoco.__version__, "numpy": np.__version__,
    }
    # Declare every seed before running; retain every attempted episode and its trace.
    save_json(output / "plan.json", plan)
    reports = []
    print(f"Frozen epoch {checkpoint['epoch']}; new seeds {seeds[0]}..{seeds[-1]}; no training.", flush=True)
    for index, seed in enumerate(seeds):
        task = VerticalGraspTask(model)
        trace = {}
        try:
            observation = task.reset(seed)
            trace, reason = run_policy(task, policy, scales, observation, None)
            result = task.result()
            finite = bool(np.isfinite(trace["final_observation"]).all()
                          and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
        except (RuntimeError, ValueError, FloatingPointError) as error:
            reason = f"execution_error: {error}"
            result = {"completed": False, "grasp_success": False,
                      "decisions": getattr(task, "decisions", 0)}
            finite = False
        unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
        commands = np.concatenate([[1.0], trace["applied_actions"][:, 1]]) if trace else np.asarray([1.0])
        report = {
            "seed": seed, "requested_xy_m": task.xy.tolist() if hasattr(task, "xy") else None,
            "stop_reason": reason, "finite_final_state": finite, **result,
            "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
            "weights_unchanged": unchanged, "gripper_command_switches": int(np.count_nonzero(np.diff(commands))),
            "trace_file": f"episode_seed{seed}.npz",
        }
        np.savez_compressed(output / report["trace_file"], **trace)
        save_json(output / f"episode_seed{seed}.json", report)
        reports.append(report)
        successes = sum(item["policy_success"] for item in reports)
        summary = {
            "schema": "vertical_grasp_evaluation_summary_v1", "checkpoint_sha256": checkpoint_hash,
            "planned_episodes": len(seeds), "attempted_episodes": len(reports),
            "completed_batch": len(reports) == len(seeds), "policy_successes": successes,
            "policy_success_rate": successes / len(reports),
            "failed_seeds": [item["seed"] for item in reports if not item["policy_success"]],
            "weights_unchanged": all(item["weights_unchanged"] for item in reports),
            "episodes": reports,
        }
        save_json(output / "summary.json", summary)
        lift = report.get("min_lift_last_hold_m")
        lift_text = f"{lift:.3f}" if lift is not None and np.isfinite(lift) else "unavailable"
        print(f"[{index + 1:02d}/{len(seeds):02d}] seed={seed} success={report['policy_success']} "
              f"min_lift={lift_text} m; {reason}", flush=True)
        if not unchanged:
            raise RuntimeError(f"Weights changed; partial records saved at {output}")
    checkpoint_unchanged = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    summary["checkpoint_file_unchanged"] = checkpoint_unchanged
    save_json(output / "summary.json", summary)
    if not checkpoint_unchanged:
        raise RuntimeError(f"Checkpoint changed during evaluation; records saved at {output}")
    print("\n--- Grasp BC evaluation ---")
    print(f"policy_successes = {successes}/{len(seeds)}")
    print(f"policy_success_rate = {summary['policy_success_rate']:.1%}")
    print(f"failed_seeds = {summary['failed_seeds']}")
    print(f"weights_unchanged = {summary['weights_unchanged']}")
    print(f"saved = {output}")
    print("Same-range new-position evaluation of prepared-hover vertical grasping; all attempts retained.")


if __name__ == "__main__":
    main()
