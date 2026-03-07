#!/usr/bin/env python3
"""
Split sweap_eval_flipt_v2.jsonl into train/val/test splits and convert to ACE format.

Splits: 50 train, 15 val, 20 test (85 total, matching flipt entry count).
Output format matches prepare_data.py (context, question, target) for use with
task_config.json swe_bench_pro_flipt.

Usage:
    python split_flipt.py
    python split_flipt.py --input data/sweap_eval_flipt_v2.jsonl --out_dir data
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

TRAIN_COUNT = 50
VAL_COUNT = 15
TEST_COUNT = 20
RANDOM_SEED = 42


def load_and_format_samples(jsonl_path: str) -> list[dict]:
    """Load SWE-bench Pro JSONL and format into ACE's {context, question, target} format."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue

            docker_image = row.get("docker_image") or row.get("image_name", "")
            fail_to_pass = row.get("fail_to_pass") or row.get("FAIL_TO_PASS", "[]")
            pass_to_pass = row.get("pass_to_pass") or row.get("PASS_TO_PASS", "[]")

            sample = {
                "context": json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "docker_image": docker_image,
                        "problem_statement": row["problem_statement"],
                        "repo": row.get("repo", ""),
                    }
                ),
                "question": row["problem_statement"],
                "target": json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "gold_patch": row.get("patch", ""),
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


def split_and_save(
    samples: list[dict],
    out_dir: str,
    train_count: int = TRAIN_COUNT,
    val_count: int = VAL_COUNT,
    test_count: int = TEST_COUNT,
    seed: int = RANDOM_SEED,
) -> dict[str, int]:
    """Shuffle, split by fixed counts, and write train/val/test JSONL files."""
    total_requested = train_count + val_count + test_count
    n = len(samples)
    if n < total_requested:
        raise ValueError(
            f"Not enough samples: have {n}, need at least {total_requested} "
            f"(train={train_count}, val={val_count}, test={test_count})"
        )

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count : train_count + val_count + test_count]

    os.makedirs(out_dir, exist_ok=True)
    outputs = {
        "train": (os.path.join(out_dir, "train_flipt.jsonl"), train),
        "val": (os.path.join(out_dir, "val_flipt.jsonl"), val),
        "test": (os.path.join(out_dir, "test_flipt.jsonl"), test),
    }
    for name, (path, data) in outputs.items():
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} samples → {path}")

    return {"train": len(train), "val": len(val), "test": len(test)}


def main() -> int:
    default_input = (
        Path(__file__).resolve().parents[1] / "data" / "sweap_eval_flipt_v2.jsonl"
    )
    default_out = Path(__file__).resolve().parents[1] / "data"

    parser = argparse.ArgumentParser(
        description="Split flipt JSONL into train/val/test (50/15/20) in ACE format"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(default_input),
        help=f"Input JSONL path (default: {default_input})",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        default=str(default_out),
        help=f"Output directory (default: {default_out})",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=TRAIN_COUNT,
        help=f"Number of train samples (default: {TRAIN_COUNT})",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=VAL_COUNT,
        help=f"Number of val samples (default: {VAL_COUNT})",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=TEST_COUNT,
        help=f"Number of test samples (default: {TEST_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Loading from {args.input}...")
    samples = load_and_format_samples(args.input)
    print(f"Loaded {len(samples)} instances. Splitting...")

    try:
        split_and_save(
            samples,
            args.out_dir,
            train_count=args.train,
            val_count=args.val,
            test_count=args.test,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
