#!/usr/bin/env python3
"""
Extract entries for the flipt-io/flipt repo from sweap_eval_full_v2.jsonl.

The repo can be identified by:
  - repo field: "flipt-io/flipt" (GitHub org/repo format)
  - image_name: contains "flipt-io.flipt" (normalized org.repo format)

Usage:
    python extract_flipt.py
    python extract_flipt.py --input /path/to/sweap_eval_full_v2.jsonl --output /path/to/sweap_eval_flipt_v2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


TARGET_REPO_GITHUB = "flipt-io/flipt"
TARGET_REPO_NORMALIZED = "flipt-io.flipt"


def is_flipt_entry(record: dict) -> bool:
    """
    Return True if the record belongs to the flipt-io/flipt repo.

    Checks:
    1. repo field equals "flipt-io/flipt"
    2. image_name contains "flipt-io.flipt" (fallback)
    """
    repo = record.get("repo", "")
    if repo == TARGET_REPO_GITHUB:
        return True

    image_name = record.get("image_name") or record.get("docker_image") or ""
    if TARGET_REPO_NORMALIZED in image_name:
        return True

    return False


def extract_flipt_entries(
    input_path: str,
    output_path: str,
    *,
    verbose: bool = True,
) -> int:
    """
    Read JSONL from input_path, filter for flipt-io/flipt entries, write to output_path.

    Returns the number of entries written.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    count = 0
    total = 0
    skipped_json = 0

    with open(input_file, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                skipped_json += 1
                continue

            if is_flipt_entry(record):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    if verbose:
        print(
            f"Extracted {count} flipt-io/flipt entries from {total} total "
            f"(skipped {skipped_json} invalid JSON lines)"
        )
        print(f"Output written to: {output_path}")

    return count


def main() -> int:
    default_input = Path(__file__).resolve().parents[1] / "data" / "sweap_eval_full_v2.jsonl"
    default_output = Path(__file__).resolve().parents[1] / "data" / "sweap_eval_flipt_v2.jsonl"

    parser = argparse.ArgumentParser(
        description="Extract flipt-io/flipt entries from sweap_eval JSONL"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(default_input),
        help=f"Input JSONL path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(default_output),
        help=f"Output JSONL path (default: {default_output})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    try:
        count = extract_flipt_entries(
            args.input,
            args.output,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
