#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASETS=(hotpotqa 2wiki musique)

RELRAG_CONFIG="${RELRAG_CONFIG:-relrag/config/config.yaml}"
TOP_K="${TOP_K:-10}"
LIMIT="${LIMIT:-}"
FORCE_BUILD=0
LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:-}"
      shift 2
      ;;
    --force-build)
      FORCE_BUILD=1
      shift
      ;;
    --top-k)
      TOP_K="${2:-10}"
      shift 2
      ;;
    --config)
      RELRAG_CONFIG="${2:-${RELRAG_CONFIG}}"
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
init_logging "relrag_hybrid" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

entry_for_dataset() {
  case "$1" in
    hotpotqa) echo "hotpot_entry.py" ;;
    2wiki) echo "twowiki_entry.py" ;;
    musique) echo "musique_entry.py" ;;
    *) return 1 ;;
  esac
}

data_for_dataset() {
  case "$1" in
    hotpotqa) echo "data/hotpot_dev_distractor_500_jsonl.jsonl" ;;
    2wiki) echo "data/2wiki_dev_sample_500.jsonl" ;;
    musique) echo "data/musique_ans_v1.0_dev_500.jsonl" ;;
    *) return 1 ;;
  esac
}

log_info "plan relrag_hybrid datasets=${#DATASETS[@]} top_k=${TOP_K}"
log_info "config=${RELRAG_CONFIG}"

TOTAL="${#DATASETS[@]}"
DONE=0

for dataset in "${DATASETS[@]}"; do
  DONE=$((DONE + 1))
  progress_bar "${DONE}" "${TOTAL}" "relrag/hybrid dataset=${dataset} (qwen+deepseek parallel)"

  entry_script="$(entry_for_dataset "${dataset}")"
  data_path="$(data_for_dataset "${dataset}")"

  CMD_BASE=(
    python "${entry_script}"
    --config "${RELRAG_CONFIG}"
    --data "${data_path}"
    --retriever hybrid
    --top_k "${TOP_K}"
    --chunking_method sentence
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD_BASE+=(--limit "${LIMIT}")
  fi
  if [[ "${FORCE_BUILD}" -eq 1 ]]; then
    CMD_BASE+=(--force_build)
  fi

  Q_TAG="relrag/hybrid/${dataset}/qwen"
  D_TAG="relrag/hybrid/${dataset}/deepseek"

  run_task_logged "${Q_TAG}" "${LOG_DIR}/relrag_${dataset}_qwen.log" \
    "${CMD_BASE[@]}" \
    --reader vllm \
    --cache_dir "result/aligned/relrag/cache/${dataset}/qwen" \
    --output_dir "result/aligned/relrag/hybrid/${dataset}/qwen" \
    --debug_dir "result/aligned/relrag/debug/${dataset}/qwen" &
  PID_Q=$!

  run_task_logged "${D_TAG}" "${LOG_DIR}/relrag_${dataset}_deepseek.log" \
    "${CMD_BASE[@]}" \
    --reader openai \
    --cache_dir "result/aligned/relrag/cache/${dataset}/deepseek" \
    --output_dir "result/aligned/relrag/hybrid/${dataset}/deepseek" \
    --debug_dir "result/aligned/relrag/debug/${dataset}/deepseek" &
  PID_D=$!

  wait_pair "${PID_Q}" "${Q_TAG}" "${PID_D}" "${D_TAG}"
done

log_info "all_done relrag_hybrid"
log_info "summary_log=${SUMMARY_LOG}"
