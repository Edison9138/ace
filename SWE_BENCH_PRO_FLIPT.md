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

## Recommended Command to Run flipt subset of SWE-Bench-Pro

```bash
uv run python -m eval.swe_bench_pro.run \
  --task_name swe_bench_pro_flipt \
  --mode offline \
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
  --traj_dir "eval/swe_bench_pro/results/flipt0/trajectories" \
  --save_dir "eval/swe_bench_pro/results/flipt0"
```

`--dockerhub_username jefzda` is where the SWE-bench Pro remote Docker images are hosted — leave this as-is.

For subsequent runs, increment the output directory index to avoid collisions:
```
# First run
--traj_dir "eval/swe_bench_pro/results/flipt0/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt0"

# Second run
--traj_dir "eval/swe_bench_pro/results/flipt1/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt1"

# Third run
--traj_dir "eval/swe_bench_pro/results/flipt2/trajectories"
--save_dir "eval/swe_bench_pro/results/flipt2"
```

---

## Key Arguments

| Argument | Description |
|---|---|
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
| `--traj_dir` | Where to save agent trajectory files |
| `--save_dir` | Where to save results and playbooks |
