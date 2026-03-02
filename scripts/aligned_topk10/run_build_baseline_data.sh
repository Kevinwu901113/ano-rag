#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_ROOT="${OUT_ROOT:-baseline/data/aligned_topk10}"
HOTPOT_PATH="${HOTPOT_PATH:-data/hotpot_dev_distractor_500_jsonl.jsonl}"
TWOWIKI_PATH="${TWOWIKI_PATH:-data/2wiki_dev_sample_500.jsonl}"
MUSIQUE_PATH="${MUSIQUE_PATH:-data/musique_ans_v1.0_dev_500.jsonl}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-32}"
LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"
CONDA_ENV="${CONDA_ENV:-${BASELINE_ENV_LIGHTRAG:-baseline-lightrag}}"

source "${SCRIPT_DIR}/lib.sh"
init_logging "build_baseline_data" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  log_info "fail conda_not_found"
  echo "conda command not found in PATH" >&2
  exit 1
fi

if ! conda run -n "${CONDA_ENV}" python -c "import sys" >/dev/null 2>&1; then
  log_info "fail invalid_conda_env=${CONDA_ENV}"
  echo "Cannot run python in conda env: ${CONDA_ENV}" >&2
  exit 1
fi

CMD=(
  conda run -n "${CONDA_ENV}" python baseline/tools/build_intermediate.py
  --out-root "${OUT_ROOT}"
  --hotpot "${HOTPOT_PATH}"
  --twowiki "${TWOWIKI_PATH}"
  --musique "${MUSIQUE_PATH}"
  --chunking_method fixed
  --chunk_size "${CHUNK_SIZE}"
  --chunk_overlap "${CHUNK_OVERLAP}"
)

log_info "build_start out_root=${OUT_ROOT} chunk=fixed/${CHUNK_SIZE}/${CHUNK_OVERLAP}"
log_info "conda_env build=${CONDA_ENV}"
run_task_logged "build_intermediate" "${LOG_DIR}/build_intermediate.log" "${CMD[@]}"
log_info "build_done"
log_info "summary_log=${SUMMARY_LOG}"
