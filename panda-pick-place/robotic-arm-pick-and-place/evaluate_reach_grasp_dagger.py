"""Paired evaluation of original BC and aggregated-label policies on new seeds."""

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
from collect_reach_grasp_dataset import save_json
from reach_grasp_task import ReachGraspTask
from rollout_reach_grasp_bc import load_policy, run_policy
from rollout_reach_grasp_dagger import digest, load_dagger_policy, post_lift_xy_displacement


def evaluate_one(model, policy, scales, seed):
    task = ReachGraspTask(model)
    trace, initial, initial_ctrl = {}, None, None
    try:
        observation = task.reset(seed)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, task.data, initial, spec)
        initial_ctrl = task.data.ctrl.copy()
        trace, reason = run_policy(task, policy, scales, observation)
        result = task.result()
        finite = bool(all(np.isfinite(value).all() for value in trace.values())
                      and np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all())
    except (RuntimeError, ValueError, FloatingPointError) as error:
        reason = f"execution_error: {error}"
        result = {"completed": False, "grasp_success": False, "decisions": getattr(task, "decisions", 0)}
        finite = False
    report = {
        "seed": seed, "stop_reason": reason, "finite_final_state": finite, **result,
        "policy_success": bool(reason == "horizon_reached" and finite and result["grasp_success"]),
        "initial_xy_error_m": getattr(task, "initial_xy_error", None),
        "cube_xy_m": task.requested_cube_xy.tolist() if hasattr(task, "requested_cube_xy") else None,
        "initial_state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest() if initial is not None else None,
    }
    if trace:
        report["maximum_cube_lift_m"] = float(np.r_[trace["observations"][:, 3], trace["final_observation"][3]].max())
        report["post_lift_cube_xy_displacement_m"] = post_lift_xy_displacement(trace)
        report["gripper_command_switches"] = int(np.count_nonzero(np.diff(np.r_[1., trace["applied_actions"][:, 3]])))
    if initial is not None:
        trace["initial_state"] = initial
        trace["initial_ctrl"] = initial_ctrl
    return report, trace, initial


def summarize(pairs, planned):
    summary = {"schema": "reach_grasp_paired_evaluation_summary_v1", "planned_pairs": planned,
               "attempted_pairs": len(pairs), "completed_batch": len(pairs) == planned,
               "paired_initial_states_match": all(p["initial_states_match"] for p in pairs), "pairs": pairs}
    for name in ("bc", "aggregated"):
        records = [pair[name] for pair in pairs]
        drift = [r["post_lift_cube_xy_displacement_m"] for r in records
                 if r.get("post_lift_cube_xy_displacement_m") is not None and np.isfinite(r["post_lift_cube_xy_displacement_m"])]
        summary[name] = {
            "successes": sum(r["policy_success"] for r in records), "attempted_episodes": len(records),
            "success_rate": sum(r["policy_success"] for r in records) / len(records),
            "failed_seeds": [r["seed"] for r in records if not r["policy_success"]],
            "weights_unchanged": all(r["weights_unchanged"] for r in records),
            "post_lift_xy_median_m": float(np.median(drift)) if drift else None,
            "post_lift_xy_metric_episodes": len(drift),
        }
    summary["improved_seeds"] = [p["seed"] for p in pairs if not p["bc"]["policy_success"] and p["aggregated"]["policy_success"]]
    summary["regressed_seeds"] = [p["seed"] for p in pairs if p["bc"]["policy_success"] and not p["aggregated"]["policy_success"]]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Aggregated-label policy; parent BC is inferred")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--label", default="new_seed_paired_bc_vs_aggregation")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Need positive --episodes and non-negative --seed.")
    root = Path(__file__).resolve().parent
    new_path = args.checkpoint.resolve()
    new_policy, new_scales, new_checkpoint, new_split, new_hash = load_dagger_policy(new_path, root)
    parent_path = Path(new_checkpoint["config"]["initial_checkpoint"])
    parent_policy, parent_scales, parent_checkpoint, parent_split, parent_hash = load_policy(parent_path, root)
    if parent_hash != new_checkpoint["config"]["initial_checkpoint_sha256"]:
        raise ValueError("Original checkpoint differs from the saved fine-tuning parent.")
    for key in ("task_source_sha256", "setup_reference_m", "control_dt_s", "physics_dt_s", "horizon_s", "max_delta_xyz_m", "reference_bounds_m"):
        if parent_checkpoint["config"][key] != new_checkpoint["config"][key]:
            raise ValueError(f"Policies use different task contracts: {key}")
    seeds = list(range(args.seed, args.seed + args.episodes))
    used = {r["seed"] for split in (parent_split, new_split) for group in ("train", "validation") for r in split[group]}
    if used.intersection(seeds):
        raise ValueError("Paired evaluation seeds overlap training or validation episodes.")
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    if model.opt.timestep != new_checkpoint["config"]["physics_dt_s"]:
        raise ValueError("Physics timestep differs from training.")
    policies = {"bc": (parent_policy, parent_scales), "aggregated": (new_policy, new_scales)}
    before = {name: [p.detach().clone() for p in policy.parameters()] for name, (policy, _) in policies.items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "runs/bc" / f"reach_grasp_paired_eval_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "reach_grasp_paired_evaluation_v1", "label": args.label, "seeds": seeds, "planned_pairs": len(seeds),
        "bc_checkpoint": str(parent_path), "bc_checkpoint_sha256": parent_hash, "bc_epoch": parent_checkpoint["epoch"],
        "aggregated_checkpoint": str(new_path), "aggregated_checkpoint_sha256": new_hash, "aggregated_epoch": new_checkpoint["epoch"],
        "training_seeds": new_checkpoint["config"]["train_seeds"], "validation_seeds": new_checkpoint["config"]["validation_seeds"],
        "seed_overlap": False, "scope": "same-range new cube positions, fixed hover start, state-based XYZ/gripper policies",
        "green_x_range_m": list(GREEN_X_RANGE), "green_y_range_m": list(GREEN_Y_RANGE),
        "setup_reference_m": ReachGraspTask.START_POSITION.tolist(),
        "criterion": "last 0.5 simulated seconds: lift >= 0.10 m, displacement <= 0.01 m, grasp-site distance <= 0.04 m",
        "extra_metric": "maximum horizontal cube displacement after first reaching 0.10 m lift; does not change success criterion",
        "expert_actions_used": False, "training_during_evaluation": False,
        "physics_dt_s": model.opt.timestep, "control_dt_s": ReachGraspTask.CONTROL_DT,
        "horizon_s": ReachGraspTask.HORIZON, "max_delta_xyz_m": ReachGraspTask.MAX_DELTA_XYZ,
        "task_source_sha256": new_checkpoint["config"]["task_source_sha256"],
        "evaluation_source_sha256": {name: digest(root / name) for name in
                                     ("evaluate_reach_grasp_dagger.py", "rollout_reach_grasp_bc.py", "rollout_reach_grasp_dagger.py")},
    }
    save_json(output / "plan.json", plan)
    pairs = []
    print(f"Paired evaluation: seeds {seeds[0]}..{seeds[-1]}; two frozen policies per seed.", flush=True)
    for index, seed in enumerate(seeds):
        pair, initial_states = {"seed": seed}, {}
        for name, (policy, scales) in policies.items():
            report, trace, initial = evaluate_one(model, policy, scales, seed)
            report["weights_unchanged"] = all(torch.equal(old, new) for old, new in zip(before[name], policy.parameters()))
            report["trace_file"] = f"{name}_seed{seed}.npz"
            np.savez_compressed(output / report["trace_file"], **trace)
            save_json(output / f"{name}_seed{seed}.json", report)
            pair[name], initial_states[name] = report, initial
        old, new = initial_states["bc"], initial_states["aggregated"]
        pair["initial_states_match"] = bool(old is not None and new is not None and old.shape == new.shape
                                            and np.allclose(old, new, rtol=0, atol=1e-10))
        pair["initial_state_max_abs_difference"] = float(np.abs(old - new).max()) if pair["initial_states_match"] else None
        pairs.append(pair)
        summary = summarize(pairs, len(seeds))
        save_json(output / "summary.json", summary)
        print(f"[{index + 1:02d}/{len(seeds):02d}] seed={seed} BC={pair['bc']['policy_success']} "
              f"aggregated={pair['aggregated']['policy_success']} same_start={pair['initial_states_match']}", flush=True)
        if not pair["initial_states_match"] or not all(pair[name]["weights_unchanged"] for name in policies):
            raise RuntimeError(f"Pairing or frozen-weight check failed; records retained at {output}")
    summary["checkpoint_files_unchanged"] = digest(parent_path) == parent_hash and digest(new_path) == new_hash
    save_json(output / "summary.json", summary)
    if not summary["checkpoint_files_unchanged"]:
        raise RuntimeError("Checkpoint files changed during evaluation.")
    print("\n--- Paired reach-and-grasp evaluation ---")
    for name in ("bc", "aggregated"):
        item = summary[name]
        print(f"{name}_successes = {item['successes']}/{len(seeds)} ({item['success_rate']:.1%})")
        print(f"{name}_failed_seeds = {item['failed_seeds']}")
        print(f"{name}_post_lift_xy_median_m = {item['post_lift_xy_median_m']}")
    print(f"regressed_seeds = {summary['regressed_seeds']}")
    print(f"paired_initial_states_match = {summary['paired_initial_states_match']}")
    print(f"weights_unchanged = {all(summary[name]['weights_unchanged'] for name in policies)}")
    print(f"saved = {output}")
    print("Both policies used the same task executor and success criterion; all outcomes retained.")


if __name__ == "__main__":
    main()
