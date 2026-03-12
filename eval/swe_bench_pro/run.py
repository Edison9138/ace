import argparse
import json
import os
import sys

# Run this script from the ace repo root: python -m eval.swe_bench_pro.run --mode ...
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ace import ACE
from ace.core import Curator, Reflector
from .swe_generator import (
    SWEBenchProGenerator,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STEP_LIMIT,
    DEFAULT_COST_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_API_PROVIDER,
)
from .data_processor import DataProcessor, load_data
from .prompts.curator_prompts import (
    SWE_CURATOR_PROMPT_NO_GT,
    SWE_CURATOR_PROMPT_WITH_GT,
)
from .prompts.reflector_prompts import SWE_REFLECTOR_PROMPT_WITH_GT, SWE_REFLECTOR_PROMPT_NO_GT


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ACE self-update loop on SWE-bench Pro with mini-swe-agent"
    )

    # ── Task / mode ───────────────────────────────────────────────────────────
    p.add_argument("--task_name", default="swe_bench_pro")
    p.add_argument("--mode", choices=["offline", "online", "eval_only"], required=True)
    p.add_argument(
        "--config_path",
        default="./eval/swe_bench_pro/data/task_config.json",
        help="Path to task_config.json with train_data/val_data/test_data keys",
    )

    # ── Model / API ───────────────────────────────────────────────────────────
    p.add_argument("--api_provider", default=DEFAULT_API_PROVIDER)
    p.add_argument("--generator_model", default=DEFAULT_MODEL)
    p.add_argument("--reflector_model", default=DEFAULT_MODEL)
    p.add_argument("--curator_model", default=DEFAULT_MODEL)
    p.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens for ACE generator/curator calls and the SWE coding agent",
    )
    p.add_argument(
        "--coding_model",
        default=DEFAULT_MODEL,
        help="LiteLLM model string for the mini-swe-agent coding loop",
    )

    # ── External data paths — user provides at runtime ────────────────────────
    p.add_argument(
        "--data_jsonl",
        default=None,
        help="Path to raw JSONL (sweap_eval_full_v2.jsonl or sweap_eval_flipt_v2.jsonl). "
        "Defaults to config 'raw_data' when task_name has it.",
    )
    p.add_argument(
        "--scripts_dir",
        default=None,
        help="Path to SWE-bench_Pro-os/run_scripts/ (optional; auto-discovers ./SWE-bench_Pro-os/run_scripts first, then falls back to GitHub per-file)",
    )
    p.add_argument(
        "--dockerfiles_dir",
        default=None,
        help="Path to SWE-bench_Pro-os/dockerfiles/ (optional; auto-discovers ./SWE-bench_Pro-os/dockerfiles first, then falls back to GitHub per-file)",
    )
    p.add_argument(
        "--dockerhub_username",
        default=os.environ.get("DOCKERHUB_USERNAME", "jefzda"),
        help="DockerHub username (defaults to DOCKERHUB_USERNAME env var or 'jefzda')",
    )

    p.add_argument(
        "--test_workers",
        type=int,
        default=1,
        help="Number of workers to use for evaluation (default: 1)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--save_dir", default="./eval/swe_bench_pro/results")
    p.add_argument("--initial_playbook", default=None)
    p.add_argument(
        "--traj_dir",
        default=None,
        help="Directory to save agent trajectory .traj.json files",
    )

    # ── Training hyper-parameters (mirrors finance/run.py) ────────────────────
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument(
        "--max_num_rounds",
        type=int,
        default=1,
        help="Max reflection rounds per sample (keep low — each round = full agent/eval run)",
    )
    p.add_argument(
        "--curator_frequency",
        type=int,
        default=1,
        help="Run curator every N training steps",
    )
    p.add_argument("--eval_steps", type=int, default=3)
    p.add_argument("--save_steps", type=int, default=1)
    # Keep playbooks compact by default; oversized playbooks can consume most
    # of the model context window before any tool interaction happens.
    p.add_argument("--playbook_token_budget", type=int, default=80000)

    # ── Data slicing (replaces the old [:3] hardcode) ─────────────────────────
    p.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Limit training to the first N samples (default: use all).",
    )
    p.add_argument(
        "--num_val_samples",
        type=int,
        default=None,
        help="Limit validation to the first N samples (default: use all).",
    )
    p.add_argument(
        "--num_test_samples",
        type=int,
        default=None,
        help="Limit test set to the first N samples (default: use all).",
    )

    # ── Agent Execution ───────────────────────────────────────────────────────
    p.add_argument("--step_limit", type=int, default=DEFAULT_STEP_LIMIT)
    p.add_argument("--cost_limit", type=float, default=DEFAULT_COST_LIMIT)

    args = p.parse_args()
    if not args.dockerhub_username:
        p.error("Please provide --dockerhub_username or set DOCKERHUB_USERNAME env var")
    return args


def load_initial_playbook(path: str) -> str | None:
    """Load initial playbook from an explicit path, falling back to the
    bundled swe_bench_pro_playbook.txt if no path is given."""
    pb_path = path or os.path.join(
        os.path.dirname(__file__), "swe_bench_pro_playbook.txt"
    )
    if os.path.exists(pb_path):
        with open(pb_path) as f:
            return f.read() or None
    return None


def preprocess_data(task_name: str, config: dict, mode: str, data_processor):
    """Load and process train/val/test splits following the standard ACE pattern.

    For offline mode: loads train + val (and optionally test).
    For online / eval_only modes: loads test only.
    """
    if mode in ("online", "eval_only"):
        if "test_data" not in config:
            raise ValueError(f"'{mode}' mode requires 'test_data' in task_config.json")
        test_samples = data_processor.process_task_data(load_data(config["test_data"]))
        train_samples = val_samples = None
        print(f"{mode} mode: {len(test_samples)} test samples")
    else:  # offline
        if "train_data" not in config or "val_data" not in config:
            raise ValueError(
                "'offline' mode requires 'train_data' and 'val_data' in task_config.json"
            )
        train_samples = data_processor.process_task_data(
            load_data(config["train_data"])
        )
        val_samples = data_processor.process_task_data(load_data(config["val_data"]))
        test_samples = (
            data_processor.process_task_data(load_data(config["test_data"]))
            if "test_data" in config
            else []
        )
        print(
            f"offline mode: {len(train_samples)} train  "
            f"{len(val_samples)} val  {len(test_samples)} test"
        )

    return train_samples, val_samples, test_samples


def main():
    args = parse_args()

    # 1. Load task config
    with open(args.config_path, "r") as f:
        task_config = json.load(f)
    config = task_config[args.task_name]

    # Resolve raw data path: explicit --data_jsonl or config "raw_data"
    data_jsonl = args.data_jsonl or config.get("raw_data")
    if not data_jsonl:
        raise ValueError(
            "No raw data path. Provide --data_jsonl or add 'raw_data' to task_config."
        )

    # 2. Build DataProcessor (needed before data loading for process_task_data)
    data_processor = DataProcessor(
        raw_samples_path=data_jsonl,
        scripts_dir=None,
        dockerfiles_dir=None,
        dockerhub_username=args.dockerhub_username,
        eval_output_dir=os.path.join(args.save_dir, "eval_outputs"),
    )

    # 3. Load and process data
    train_samples, val_samples, test_samples = preprocess_data(
        args.task_name, config, args.mode, data_processor
    )

    # 4. Optional data slicing to run
    if train_samples is not None and args.num_train_samples is not None:
        train_samples = train_samples[: args.num_train_samples]
        print(f"  → using first {len(train_samples)} train samples")
    if val_samples is not None and args.num_val_samples is not None:
        val_samples = val_samples[: args.num_val_samples]
        print(f"  → using first {len(val_samples)} val samples")
    if test_samples is not None and args.num_test_samples is not None:
        test_samples = test_samples[: args.num_test_samples]
        print(f"  → using first {len(test_samples)} test samples")

    # 5. Load initial playbook
    initial_playbook = load_initial_playbook(args.initial_playbook)
    if initial_playbook:
        src = args.initial_playbook or "swe_bench_pro_playbook.txt (bundled)"
        print(f"Loaded initial playbook from {src}")
    else:
        print("No initial playbook found — starting with empty playbook")

    # 6. Build ACE system
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
    )

    ace_system.generator = SWEBenchProGenerator(
        api_client=ace_system.generator_client,
        api_provider=args.api_provider,
        model=args.generator_model,
        coding_model_name=args.coding_model,
        dockerhub_username=args.dockerhub_username,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        max_tokens=args.max_tokens,
        traj_output_dir=args.traj_dir,
    )

    # 7b. Replace reflector with SWE-specific one.
    #
    # The SWE reflector differs from the default in two ways:
    #   1. It receives the full playbook as context (not just cited bullets) so it
    #      can write analysis that references existing bullets and avoid recommending
    #      what is already covered — mirroring ace-appworld's approach.
    #   2. Its output schema drops `bullet_tags` since mini-swe-agent never cites
    #      bullet IDs inline (it outputs a git-diff patch), making per-bullet tagging
    #      impossible.
    ace_system.reflector = Reflector(
        api_client=ace_system.reflector_client,
        api_provider=args.api_provider,
        model=args.reflector_model,
        max_tokens=args.max_tokens,
        prompt_with_gt=SWE_REFLECTOR_PROMPT_WITH_GT,
        prompt_no_gt=SWE_REFLECTOR_PROMPT_NO_GT,
    )

    # 7c. Replace curator with SWE-specific one.
    ace_system.curator = Curator(
        api_client=ace_system.curator_client,
        api_provider=args.api_provider,
        model=args.curator_model,
        max_tokens=args.max_tokens,
        prompt_with_gt=SWE_CURATOR_PROMPT_WITH_GT,
        prompt_no_gt=SWE_CURATOR_PROMPT_NO_GT,
    )

    # 8. Run
    run_config = {
        "task_name": args.task_name,
        "save_dir": args.save_dir,
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "json_mode": True,
        "test_workers": args.test_workers,
        "playbook_token_budget": args.playbook_token_budget,
        "max_tokens": args.max_tokens,
        "coding_model": args.coding_model,
        "step_limit": args.step_limit,
        "cost_limit": args.cost_limit,
        "dockerhub_username": args.dockerhub_username,
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=run_config,
    )
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()
