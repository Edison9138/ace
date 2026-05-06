#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh [smoke|pilot|full|scan_aug] [run_name]
#
# Examples:
#   bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh smoke flipt_batch_smoke0
#   bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh pilot flipt_batch_pilot0
#   bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh full flipt_batch0
#   bash eval/swe_bench_pro/scripts/run_flipt_acebatch.sh scan_aug flipt_scan_aug0

PROFILE="${1:-full}"
RUN_NAME="${2:-flipt_batch_${PROFILE}0}"
SCAN_AUG=false

case "${PROFILE}" in
  smoke)
    NUM_TRAIN_SAMPLES=1
    NUM_VAL_SAMPLES=1
    NUM_TEST_SAMPLES=1
    TEST_WORKERS=1
    EVAL_STEPS=1
    SAVE_STEPS=1
    STEP_LIMIT=40
    COST_LIMIT=3.0
    BATCH_SIZE=1
    CURATOR_BATCH_SIZE=1
    ;;
  pilot)
    NUM_TRAIN_SAMPLES=5
    NUM_VAL_SAMPLES=5
    NUM_TEST_SAMPLES=5
    TEST_WORKERS=5
    EVAL_STEPS=5
    SAVE_STEPS=1
    STEP_LIMIT=80
    COST_LIMIT=10.0
    BATCH_SIZE=2
    CURATOR_BATCH_SIZE=2
    ;;
  full)
    NUM_TRAIN_SAMPLES=""
    NUM_VAL_SAMPLES=""
    NUM_TEST_SAMPLES=""
    TEST_WORKERS=20
    EVAL_STEPS=5
    SAVE_STEPS=1
    STEP_LIMIT=80
    COST_LIMIT=10.0
    BATCH_SIZE=2
    CURATOR_BATCH_SIZE=2
    ;;
  scan_aug)
    NUM_TRAIN_SAMPLES=""
    NUM_VAL_SAMPLES=""
    NUM_TEST_SAMPLES=""
    TEST_WORKERS=20
    EVAL_STEPS=5
    SAVE_STEPS=1
    STEP_LIMIT=80
    COST_LIMIT=10.0
    BATCH_SIZE=30
    # Leave unset so --scan_aug derives round(sqrt(30)) = 5.
    CURATOR_BATCH_SIZE=""
    SCAN_AUG=true
    ;;
  *)
    echo "Invalid profile: ${PROFILE}" >&2
    echo "Use one of: smoke, pilot, full, scan_aug" >&2
    exit 1
    ;;
esac

SAVE_DIR="eval/swe_bench_pro/results/${RUN_NAME}"
TRAJ_DIR="${SAVE_DIR}/trajectories"

if ! mkdir "${SAVE_DIR}"; then
  echo "Refusing to reuse existing result directory: ${SAVE_DIR}" >&2
  echo "Choose a new run_name to avoid contaminating old eval_outputs/trajectories." >&2
  exit 1
fi

ARGS=(
  --task_name swe_bench_pro_flipt
  --mode offline
  --ace_backend ace_batch
  --data_jsonl "eval/swe_bench_pro/data/sweap_eval_flipt_v2.jsonl"
  --config_path "eval/swe_bench_pro/data/task_config.json"
  --initial_playbook "eval/swe_bench_pro/playbooks/swe_bench_pro_flipt_playbook.txt"
  --dockerhub_username jefzda
  --num_epochs 1
  --max_num_rounds 1
  --curator_frequency 1
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --playbook_token_budget 80000
  --test_workers "${TEST_WORKERS}"
  --step_limit "${STEP_LIMIT}"
  --cost_limit "${COST_LIMIT}"
  --batch_size "${BATCH_SIZE}"
  --continue_on_llm_error
  --traj_dir "${TRAJ_DIR}"
  --save_dir "${SAVE_DIR}"
)

if [[ -n "${CURATOR_BATCH_SIZE}" ]]; then
  ARGS+=(--curator_batch_size "${CURATOR_BATCH_SIZE}")
fi
if [[ "${SCAN_AUG}" == "true" ]]; then
  ARGS+=(--scan_aug --skip_post_curate_generation)
fi
if [[ -n "${NUM_TRAIN_SAMPLES}" ]]; then
  ARGS+=(--num_train_samples "${NUM_TRAIN_SAMPLES}")
fi
if [[ -n "${NUM_VAL_SAMPLES}" ]]; then
  ARGS+=(--num_val_samples "${NUM_VAL_SAMPLES}")
fi
if [[ -n "${NUM_TEST_SAMPLES}" ]]; then
  ARGS+=(--num_test_samples "${NUM_TEST_SAMPLES}")
fi

echo "Running profile=${PROFILE} run_name=${RUN_NAME}"
echo "save_dir=${SAVE_DIR}"
echo "traj_dir=${TRAJ_DIR}"

uv run python -m eval.swe_bench_pro.run "${ARGS[@]}"
