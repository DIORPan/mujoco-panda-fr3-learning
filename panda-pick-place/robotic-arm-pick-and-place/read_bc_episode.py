"""Read one sample from a saved BC episode; does not run physics or write files."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=1500, help="Zero-based sample index")
    parser.add_argument("--episode", type=Path, help="NPZ file; defaults to the newest saved episode")
    args = parser.parse_args()

    path = args.episode
    if path is None:
        directory = Path(__file__).resolve().parent / "datasets/bc"
        episodes = list(directory.rglob("green_*.npz"))
        if not episodes:
            parser.error(f"No episodes found in {directory}; run collect_bc_episode.py first.")
        path = max(episodes, key=lambda candidate: candidate.stat().st_mtime_ns)

    with np.load(path, allow_pickle=False) as saved:
        observations = saved["observations"]
        actions = saved["actions"]
        times = saved["times"]
        if observations.ndim != 2 or observations.shape[1] != 25 or actions.shape != (len(observations), 8):
            parser.error("Expected observations (N, 25) and actions (N, 8).")
        if times.shape != (len(observations),):
            parser.error("Expected one timestamp per observation.")
        row = args.row
        if not 0 <= row < len(observations):
            parser.error(f"--row must be between 0 and {len(observations) - 1}.")

        np.set_printoptions(precision=6, suppress=True)
        print(f"file = {path.name}")
        print(f"observations.shape = {observations.shape}; actions.shape = {actions.shape}")
        print(f"all finite = {np.isfinite(observations).all() and np.isfinite(actions).all()}")
        print(f"row = {row} (zero-based); elapsed = {times[row] - times[0]:.3f} s")

        # Schema v1 columns 14 and 15 hold actual finger joint positions.
        # The last action column holds the expert's normalized gripper command.
        print(f"observations[row, 14:16] = {observations[row, 14:16]} m (actual fingers BEFORE action)")
        print(f"actions[row, :7] = {actions[row, :7]} rad (arm targets)")
        print(f"actions[row, 7] = {actions[row, 7]:.1f} (1=open, 0=close)")
        if row > 0:
            print(f"previous gripper command = {actions[row - 1, 7]:.1f}")
        print(f"cube position = {observations[row, 19:22]} m")
        print(f"target position = {observations[row, 22:25]} m")
        print(f"expert_success = {bool(saved['expert_success'])} (original criterion)")
        print("Read-only inspection. No physics run and no policy training.")


if __name__ == "__main__":
    main()
