#!/usr/bin/env python3
"""
Run once before training to generate train/val/test splits from the source JSONL.

Usage:
    python prepare_data.py \
        --data_jsonl /path/to/SWE-bench_Pro-os/helper_code/sweap_eval_full_v2.jsonl
"""

import argparse
import json
import os
import random
import pandas as pd

TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def load_and_format_samples(jsonl_path: str) -> list[dict]:
    """Load SWE-bench Pro JSONL and format into ACE's {context, question, target} format."""
    df = pd.read_json(jsonl_path, lines=True).fillna("")
    samples = []
    for _, row in df.iterrows():
        # Normalize field names (JSONL uses uppercase FAIL_TO_PASS / image_name)
        docker_image = row.get("docker_image") or row.get("image_name", "")
        fail_to_pass = row.get("fail_to_pass") or row.get("FAIL_TO_PASS", "[]")
        pass_to_pass = row.get("pass_to_pass") or row.get("PASS_TO_PASS", "[]")

        sample = {
            # context: metadata the generator needs to launch the correct SWE task image
            "context": json.dumps(
                {
                    "instance_id": row["instance_id"],
                    "docker_image": docker_image,
                    "problem_statement": row["problem_statement"],
                    "repo": row.get("repo", ""),
                }
            ),
            # question: plain text for ACE's reflector/curator
            "question": row["problem_statement"],
            # target: metadata answer_is_correct needs to run the SWE eval harness
            "target": json.dumps(
                {
                    "instance_id": row["instance_id"],
                    "gold_patch": row.get("patch", ""),
                    # Store as proper JSON arrays (not Python repr) so that
                    # downstream code can use json.loads() instead of eval().
                    "fail_to_pass": json.dumps(
                        fail_to_pass
                        if isinstance(fail_to_pass, list)
                        else json.loads(fail_to_pass)
                    ),
                    "pass_to_pass": json.dumps(
                        pass_to_pass
                        if isinstance(pass_to_pass, list)
                        else json.loads(pass_to_pass)
                    ),
                }
            ),
        }
        samples.append(sample)
    return samples


def split_and_save(samples: list[dict], out_dir: str, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VAL_FRACTION)

    splits = {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples → {path}")

    # Note: Omitted task_config.json regeneration as it overwrites the configured keys.


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_jsonl",
        required=True,
        help="Path to sweap_eval_full_v2.jsonl from SWE-bench_Pro-os",
    )
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    print(f"Loading from {args.data_jsonl}...")
    samples = load_and_format_samples(args.data_jsonl)
    print(f"Loaded {len(samples)} instances. Splitting...")
    split_and_save(samples, args.out_dir)
