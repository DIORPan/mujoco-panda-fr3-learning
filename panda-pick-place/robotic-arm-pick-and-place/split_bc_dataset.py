"""Split one collected batch by complete episodes into train and validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def make_split(manifest_path, split_seed, validation_episodes):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "panda_green_bc_batch_v1":
        raise ValueError("Expected a panda_green_bc_batch_v1 manifest.")
    records = manifest["episodes"]
    if len(records) != manifest["requested_episodes"]:
        raise ValueError("Batch is incomplete: episode count does not match manifest.")
    if not 1 <= validation_episodes < len(records):
        raise ValueError("Reserve at least one episode for each split.")
    if len({record["file"] for record in records}) != len(records):
        raise ValueError("Duplicate episode files in manifest.")
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Duplicate episode seeds in manifest.")

    entries = []
    reference_sources = None
    for record in records:
        name = record["file"]
        if Path(name).name != name:
            raise ValueError("Episode filenames must refer to files in the batch folder.")
        episode_path = manifest_path.parent / name
        with np.load(episode_path, allow_pickle=False) as saved:
            observations = saved["observations"]
            actions = saved["actions"]
            times = saved["times"]
            count = record["samples"]
            if observations.shape != (count, 25) or actions.shape != (count, 8):
                raise ValueError(f"Invalid array shapes: {name}")
            if count < 2 or times.shape != (count,):
                raise ValueError(f"Invalid episode length or timestamps: {name}")
            if not (np.isfinite(observations).all() and np.isfinite(actions).all() and np.isfinite(times).all()):
                raise ValueError(f"Non-finite values: {name}")
            if not ((actions[:, 7] >= 0).all() and (actions[:, 7] <= 1).all()):
                raise ValueError(f"Gripper labels outside [0, 1]: {name}")
            metadata = json.loads(str(saved["metadata_json"]))
            if metadata["schema"] != "panda_green_bc_v1" or metadata["observation_widths"] != [7, 7, 2, 3, 3, 3]:
                raise ValueError(f"Unexpected observation schema: {name}")
            if metadata["seed"] != record["seed"]:
                raise ValueError(f"Seed does not match manifest: {name}")
            if not np.allclose(np.diff(times), metadata["timestep_s"], rtol=0, atol=1e-10):
                raise ValueError(f"Invalid sample spacing: {name}")
            if not bool(saved["expert_success"]) or not record["expert_success"]:
                raise ValueError(f"This initial BC split requires successful expert demos; failed episode remains saved: {name}")
            if reference_sources is None:
                reference_sources = metadata["source_sha256"]
            elif metadata["source_sha256"] != reference_sources:
                raise ValueError(f"Episode source versions differ within batch: {name}")
        entries.append({
            "file": name,
            "seed": record["seed"],
            "samples": count,
            "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
        })

    if sum(entry["samples"] for entry in entries) != manifest["total_samples"]:
        raise ValueError("Sample counts do not match the manifest total.")

    # Shuffle EPISODES, not individual rows from the trajectories.
    order = np.random.default_rng(split_seed).permutation(len(entries))
    validation = [entries[int(index)] for index in order[:validation_episodes]]
    train = [entries[int(index)] for index in order[validation_episodes:]]
    return {
        "schema": "panda_green_bc_split_v1",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_seed": split_seed,
        "unit": "whole_episode",
        "train": train,
        "validation": validation,
        "train_samples": sum(entry["samples"] for entry in train),
        "validation_samples": sum(entry["samples"] for entry in validation),
        "note": "Validation measures held-out demonstrations; closed-loop policy testing is a separate step.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Exact batch manifest to split")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=2)
    args = parser.parse_args()
    if args.split_seed < 0:
        parser.error("--split-seed must be non-negative.")
    manifest_path = args.manifest.resolve()
    try:
        split = make_split(manifest_path, args.split_seed, args.validation_episodes)
        output = manifest_path.parent / f"split_seed{args.split_seed}_val{args.validation_episodes}.json"
        if output.exists():
            if json.loads(output.read_text(encoding="utf-8")) != split:
                raise ValueError(f"Existing split differs; inspect it before proceeding: {output}")
            print("Reusing existing identical split.")
        else:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(split, stream, ensure_ascii=False, indent=2)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("--- Dataset split ---")
    print(f"train: {len(split['train'])} episodes, {split['train_samples']} samples")
    print(f"validation: {len(split['validation'])} episodes, {split['validation_samples']} samples")
    print(f"train seeds = {[entry['seed'] for entry in split['train']]}")
    print(f"validation seeds = {[entry['seed'] for entry in split['validation']]}")
    print(f"saved = {output}")
    print("Whole episodes kept separate. No policy trained.")


if __name__ == "__main__":
    main()
