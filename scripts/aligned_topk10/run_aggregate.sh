#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASELINE_ROOT="${BASELINE_ROOT:-baseline/results/aligned_topk10}"
RELRAG_ROOT="${RELRAG_ROOT:-result/aligned/relrag/hybrid}"
OUTPUT_DIR="${OUTPUT_DIR:-result/aligned/comparison_topk10}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"

GOLD_HOTPOT="${GOLD_HOTPOT:-data/hotpot_dev_distractor_500_jsonl.jsonl}"
GOLD_2WIKI="${GOLD_2WIKI:-data/2wiki_dev_sample_500.jsonl}"
GOLD_MUSIQUE="${GOLD_MUSIQUE:-data/musique_ans_v1.0_dev_500.jsonl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-root)
      BASELINE_ROOT="${2:-${BASELINE_ROOT}}"
      shift 2
      ;;
    --relrag-root)
      RELRAG_ROOT="${2:-${RELRAG_ROOT}}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-${OUTPUT_DIR}}"
      shift 2
      ;;
    --expected-count)
      EXPECTED_COUNT="${2:-${EXPECTED_COUNT}}"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="${2:-${LOG_ROOT}}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

source "${SCRIPT_DIR}/lib.sh"
init_logging "aggregate" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

CMD=(
  python scripts/aggregate_aligned_results.py
  --baseline_root "${BASELINE_ROOT}"
  --relrag_root "${RELRAG_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --gold_hotpot "${GOLD_HOTPOT}"
  --gold_2wiki "${GOLD_2WIKI}"
  --gold_musique "${GOLD_MUSIQUE}"
  --expected_count "${EXPECTED_COUNT}"
)

run_task_logged "aggregate" "${LOG_DIR}/aggregate.log" "${CMD[@]}"
log_info "aggregate_done output_dir=${OUTPUT_DIR}"
log_info "summary_log=${SUMMARY_LOG}"
