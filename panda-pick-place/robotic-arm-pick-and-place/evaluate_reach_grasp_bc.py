"""Evaluate a frozen XYZ-and-gripper policy on a declared batch of new seeds."""

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
from collect_reach_grasp_dataset import json_safe
from reach_grasp_task import ReachGraspTask
from rollout_reach_grasp_bc import load_policy, run_policy


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--label", default="unseen_seed_reach_grasp_evaluation")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Need positive --episodes and non-negative --seed.")
    root = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve()
    policy, scales, checkpoint, split, checkpoint_hash = load_policy(checkpoint_path, root)
    config = checkpoint["config"]
    trainer_hash = hashlib.sha256((root / "train_reach_grasp_bc.py").read_bytes()).hexdigest()
    if trainer_hash != config["trainer_sha256"]:
        raise ValueError("Policy implementation changed since training.")
    seeds = list(range(args.seed, args.seed + args.episodes))
    used_seeds = {record["seed"] for group in ("train", "validation") for record in split[group]}
    if used_seeds.intersection(seeds):
        raise ValueError("Evaluation seeds overlap training or validation; choose new seeds.")
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    if model.opt.timestep != config["physics_dt_s"]:
        raise ValueError("Physics timestep differs from training.")
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_eval_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "reach_grasp_evaluation_v1", "label": args.label,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash, "epoch": checkpoint["epoch"],
        "seeds": seeds, "planned_episodes": len(seeds), "seed_overlap": False,
        "training_seeds": config["train_seeds"], "validation_seeds": config["validation_seeds"],
        "green_x_range_m": list(GREEN_X_RANGE), "green_y_range_m": list(GREEN_Y_RANGE),
        "setup_reference_m": ReachGraspTask.START_POSITION.tolist(),
        "scope": "same-range new positions; fixed hover start; learned XYZ approach, grasp and lift",
        "control": "network chooses all four task actions; IK keeps downward gripper orientation",
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "control_dt_s": ReachGraspTask.CONTROL_DT, "horizon_s": ReachGraspTask.HORIZON,
        "max_delta_xyz_m": ReachGraspTask.MAX_DELTA_XYZ, "physics_dt_s": model.opt.timestep,
        "expert_actions_used_during_task": False, "task_source_sha256": config["task_source_sha256"],
        "evaluation_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                     for name in ("evaluate_reach_grasp_bc.py", "rollout_reach_grasp_bc.py", "train_reach_grasp_bc.py")},
        "torch": str(torch.__version__), "mujoco": mujoco.__version__, "numpy": np.__version__,
    }
    save_json(output / "plan.json", plan)
    reports = []
    print(f"Frozen reach-and-grasp epoch {checkpoint['epoch']}; new seeds {seeds[0]}..{seeds[-1]}; no training.", flush=True)
    for index, seed in enumerate(seeds):
        task = ReachGraspTask(model)
        trace = {}
        try:
            observation = task.reset(seed)
            trace, reason = run_policy(task, policy, scales, observation)
            result = task.result()
            finite = bool(np.isfinite(trace["final_observation"]).all()
                          and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
        except (RuntimeError, ValueError, FloatingPointError) as error:
            reason = f"execution_error: {error}"
            result = {"completed": False, "grasp_success": False, "decisions": getattr(task, "decisions", 0)}
            finite = False
        unchanged = all(torch.equal(old, new) for old, new in zip(before, policy.parameters()))
        commands = np.concatenate([[1.], trace["applied_actions"][:, 3]]) if trace else np.asarray([1.])
        report = {
            "seed": seed, "requested_cube_xy_m": task.requested_cube_xy.tolist() if hasattr(task, "requested_cube_xy") else None,
            "initial_xy_error_m": getattr(task, "initial_xy_error", None),
            "stop_reason": reason, "finite_final_state": finite, **result,
            "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
            "weights_unchanged": unchanged, "gripper_command_switches": int(np.count_nonzero(np.diff(commands))),
            "trace_file": f"episode_seed{seed}.npz",
        }
        if trace:
            lifts = np.concatenate([trace["observations"][:, 3], [trace["final_observation"][3]]])
            report["maximum_cube_lift_m"] = float(lifts.max())
            report["final_xy_error_m"] = float(np.linalg.norm(trace["final_observation"][:2]))
        np.savez_compressed(output / report["trace_file"], **trace)
        save_json(output / f"episode_seed{seed}.json", report)
        reports.append(report)
        successes = sum(item["policy_success"] for item in reports)
        summary = {
            "schema": "reach_grasp_evaluation_summary_v1", "checkpoint_sha256": checkpoint_hash,
            "planned_episodes": len(seeds), "attempted_episodes": len(reports),
            "completed_batch": len(reports) == len(seeds), "policy_successes": successes,
            "policy_success_rate": successes / len(reports),
            "failed_seeds": [item["seed"] for item in reports if not item["policy_success"]],
            "weights_unchanged": all(item["weights_unchanged"] for item in reports), "episodes": reports,
        }
        save_json(output / "summary.json", summary)
        lift = report.get("min_lift_last_hold_m")
        lift_text = f"{lift:.3f}" if lift is not None and np.isfinite(lift) else "unavailable"
        print(f"[{index + 1:02d}/{len(seeds):02d}] seed={seed} success={report['policy_success']} "
              f"min_lift={lift_text} m; {reason}", flush=True)
        if not unchanged:
            raise RuntimeError(f"Weights changed; partial records saved at {output}")
    summary["checkpoint_file_unchanged"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    save_json(output / "summary.json", summary)
    if not summary["checkpoint_file_unchanged"]:
        raise RuntimeError(f"Checkpoint file changed during evaluation; records saved at {output}")
    print("\n--- Reach-and-grasp BC evaluation ---")
    print(f"policy_successes = {successes}/{len(seeds)}")
    print(f"policy_success_rate = {summary['policy_success_rate']:.1%}")
    print(f"failed_seeds = {summary['failed_seeds']}")
    print(f"weights_unchanged = {summary['weights_unchanged']}")
    print(f"saved = {output}")
    print("Fixed-start XYZ reach-and-grasp on same-range new positions; all attempts retained.")


if __name__ == "__main__":
    main()
