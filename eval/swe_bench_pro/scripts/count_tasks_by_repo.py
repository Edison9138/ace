#!/usr/bin/env python3
"""
Extract normalized repo names from sweap_eval_full_v2.jsonl and count tasks per repo.

Normalized repo format: org/repo -> org.repo (lowercase, slash replaced with dot)
Example: NodeBB/NodeBB -> nodebb.nodebb
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def normalize_repo(repo: str) -> str:
    """Convert org/repo to org.repo (lowercase)."""
    if not repo or "/" not in repo:
        return ""
    return repo.lower().replace("/", ".")


def extract_and_count(jsonl_path: str) -> Counter[str]:
    """Read JSONL, extract normalized repos, return counts per repo."""
    counts: Counter[str] = Counter()
    seen_instance_ids: set[str] = set()
    skipped = 0

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                skipped += 1
                continue

            # Prefer 'repo' field; fallback: derive from image_name
            repo = record.get("repo", "")
            if not repo:
                img = record.get("image_name") or record.get("docker_image") or ""
                # Extract from image_name: .../sweap-images/org.repo:tag
                if "sweap-images/" in img and ":" in img:
                    prefix, _ = img.split(":", 1)
                    part = prefix.split("sweap-images/")[-1].strip()
                    if part:
                        # org.repo -> org/repo for normalize_repo
                        repo = part.replace(".", "/", 1)
            if not repo:
                skipped += 1
                continue

            norm = normalize_repo(repo)
            if not norm:
                skipped += 1
                continue

            instance_id = record.get("instance_id", "")
            if instance_id and instance_id in seen_instance_ids:
                # Dedupe by instance_id (one task per instance)
                continue
            if instance_id:
                seen_instance_ids.add(instance_id)

            counts[norm] += 1

    if skipped:
        print(f"Warning: Skipped {skipped} records", file=sys.stderr)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count tasks per normalized repo in sweap_eval JSONL"
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sweap_eval_full_v2.jsonl"),
        help="Path to sweap_eval JSONL file",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "count"],
        default="count",
        help="Sort output by repo name or by count (default: count)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order",
    )
    args = parser.parse_args()

    counts = extract_and_count(args.jsonl)

    if not counts:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    items = list(counts.items())
    if args.sort == "name":
        items.sort(key=lambda x: x[0], reverse=args.reverse)
    else:
        items.sort(key=lambda x: (x[1], x[0]), reverse=not args.reverse)

    total = sum(counts.values())
    print(f"# Normalized repo -> task count (total: {total} tasks across {len(counts)} repos)\n")
    for norm_repo, count in items:
        print(f"{norm_repo}\t{count}")


if __name__ == "__main__":
    main()
