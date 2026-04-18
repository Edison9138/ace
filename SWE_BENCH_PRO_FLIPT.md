# Running SWE-bench Pro (Flipt Subset)

## Prerequisites

### 1. OpenAI API Key
```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Modal Setup
Install and authenticate:
```bash
pip install modal
modal setup   # opens browser for auth
```

> Modal gives $5 in free credits by default. Adding a credit card unlocks an additional $25 in free credits. **This eval costs ~$10 and ~12 hours to complete** — keep your laptop connected and awake for the duration.

---

## Recommended Commands for Flipt

### A) Original ACE baseline (reference)

```bash
uv run python -m eval.swe_bench_pro.run \
  --task_name swe_bench_pro_flipt \
  --mode offline \
  --ace_backend ace \
  --data_jsonl "eval/swe_bench_pro/data/sweap_eval_flipt_v2.jsonl" \
  --config_path "eval/swe_bench_pro/data/task_config.json" \
  --initial_playbook "eval/swe_bench_pro/playbooks/swe_bench_pro_flipt_playbook.txt" \
  --dockerhub_username jefzda \
  --num_epochs 1 \
  --max_num_rounds 1 \
  --curator_frequency 1 \
  --eval_steps 5 \
  --save_steps 1 \
  --playbook_token_budget 80000 \
  --test_workers 20 \
  --step_limit 80 \
  --cost_limit 10.0 \
  --traj_dir "eval/swe_bench_pro/results/flipt0/trajectories" \
  --save_dir "eval/swe_bench_pro/results/flipt0"
```

`--dockerhub_username jefzda` is where the SWE-bench Pro remote Docker images are hosted — leave this as-is.

### B) ACEBatch suitable configs (recommended)

The easiest way to avoid debug-limit mistakes is to use the preset script:

```bash
bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh [smoke|pilot|full] [run_name]
```

Examples:

```bash
# 1/1/1 quick sanity pass
bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh smoke flipt_batch_smoke0

# 5/5/5 pilot comparison run
bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh pilot flipt_batch_pilot0

# Full flipt run
bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh full flipt_batch0
```

Preset behavior:

| Profile | Train/Val/Test samples | Batch config | Workers | Step/Cost limits |
|---|---|---|---|---|
| `smoke` | `1/1/1` | `batch_size=1`, `curator_batch_size=1` | `1` | `step_limit=40`, `cost_limit=3.0` |
| `pilot` | `5/5/5` | `batch_size=2`, `curator_batch_size=2` | `5` | `step_limit=80`, `cost_limit=10.0` |
| `full` | all | `batch_size=2`, `curator_batch_size=2` | `20` | `step_limit=80`, `cost_limit=10.0` |

All presets keep your original Flipt command settings and add:
- `--ace_backend ace_batch`
- `--continue_on_llm_error`
- explicit non-debug `--step_limit/--cost_limit`

For manual ACEBatch runs (without the script), this is the full-profile command:

```bash
uv run python -m eval.swe_bench_pro.run \
  --task_name swe_bench_pro_flipt \
  --mode offline \
  --ace_backend ace_batch \
  --data_jsonl "eval/swe_bench_pro/data/sweap_eval_flipt_v2.jsonl" \
  --config_path "eval/swe_bench_pro/data/task_config.json" \
  --initial_playbook "eval/swe_bench_pro/playbooks/swe_bench_pro_flipt_playbook.txt" \
  --dockerhub_username jefzda \
  --num_epochs 1 \
  --max_num_rounds 1 \
  --curator_frequency 1 \
  --eval_steps 5 \
  --save_steps 1 \
  --playbook_token_budget 80000 \
  --test_workers 20 \
  --step_limit 80 \
  --cost_limit 10.0 \
  --batch_size 2 \
  --curator_batch_size 2 \
  --continue_on_llm_error \
  --traj_dir "eval/swe_bench_pro/results/flipt_batch0/trajectories" \
  --save_dir "eval/swe_bench_pro/results/flipt_batch0"
```

For subsequent runs, increment output directory indexes to avoid collisions:
```
# ACE baseline examples
--traj_dir "eval/swe_bench_pro/results/flipt0/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt0"

# ACEBatch examples
--traj_dir "eval/swe_bench_pro/results/flipt_batch0/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt_batch0"

--traj_dir "eval/swe_bench_pro/results/flipt_batch1/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt_batch1"
```

---

## Key Arguments

| Argument | Description |
|---|---|
| `--ace_backend` | Orchestrator backend (`ace` or `ace_batch`) |
| `--data_jsonl` | Path to the flipt JSONL dataset |
| `--config_path` | Task config with train/val/test split paths |
| `--initial_playbook` | Path to the Flipt-specific starting playbook |
| `--dockerhub_username` | DockerHub account hosting SWE-bench Pro images (`jefzda`) |
| `--num_epochs` | Training epochs |
| `--max_num_rounds` | Max reflection rounds per sample (each = full agent + eval run) |
| `--curator_frequency` | Run curator every N training steps |
| `--eval_steps` | Evaluate every N steps |
| `--save_steps` | Save checkpoint every N steps |
| `--playbook_token_budget` | Max tokens for playbook (keep ≤80000 to avoid crowding context) |
| `--test_workers` | Parallel Modal workers for evaluation |
| `--num_train_samples` | Limit training samples (omit to use all) |
| `--num_val_samples` | Limit validation samples (omit to use all) |
| `--num_test_samples` | Limit test samples (omit to use all) |
| `--step_limit` | Max mini-swe-agent tool turns per instance |
| `--cost_limit` | Max mini-swe-agent budget per instance |
| `--batch_size` | ACEBatch generator/reflector batch size |
| `--curator_batch_size` | ACEBatch curator chunk size |
| `--continue_on_llm_error` | Continue on recoverable LLM/API errors |
| `--traj_dir` | Where to save agent trajectory files |
| `--save_dir` | Where to save results and playbooks |
