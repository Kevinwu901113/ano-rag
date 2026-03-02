#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
METHODS=(bm25 dense lightrag graphrag raptor)

LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"
LIMIT="${LIMIT:-}"
RETRIEVAL_ONLY=0
REBUILD_INDEX=0
TOP_K="${TOP_K:-10}"
TIMEOUT="${TIMEOUT:-180}"
PARALLEL_METHODS="${PARALLEL_METHODS:-1}"
DATA_ROOT="${DATA_ROOT:-baseline/data/aligned_topk10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-baseline/results/aligned_topk10}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-baseline/workspaces/aligned_topk10}"
RELRAG_CONFIG="${RELRAG_CONFIG:-relrag/config/config.yaml}"
GRAPHRAG_PARALLEL_DATASETS="${GRAPHRAG_PARALLEL_DATASETS:-0}"
GRAPHRAG_QA_MODE="${GRAPHRAG_QA_MODE:-}"
COMMUNITY_REPORT_WORKFLOW="${COMMUNITY_REPORT_WORKFLOW:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:-}"
      shift 2
      ;;
    --retrieval-only)
      RETRIEVAL_ONLY=1
      shift
      ;;
    --rebuild-index)
      REBUILD_INDEX=1
      shift
      ;;
    --top-k)
      TOP_K="${2:-10}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-180}"
      shift 2
      ;;
    --parallel-methods)
      PARALLEL_METHODS="${2:-1}"
      shift 2
      ;;
    --graphrag-parallel-datasets)
      GRAPHRAG_PARALLEL_DATASETS=1
      shift
      ;;
    --graphrag_qa_mode|--graphrag-qa-mode)
      GRAPHRAG_QA_MODE="${2:-}"
      shift 2
      ;;
    --community_report_workflow|--community-report-workflow)
      COMMUNITY_REPORT_WORKFLOW="${2:-}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:-${DATA_ROOT}}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-${OUTPUT_ROOT}}"
      shift 2
      ;;
    --workspace-root)
      WORKSPACE_ROOT="${2:-${WORKSPACE_ROOT}}"
      shift 2
      ;;
    --relrag-config)
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
init_logging "baseline_all" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

if ! [[ "${PARALLEL_METHODS}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL_METHODS}" -lt 1 ]]; then
  echo "--parallel-methods must be a positive integer, got: ${PARALLEL_METHODS}" >&2
  exit 2
fi

TOTAL="${#METHODS[@]}"
run_one_method() {
  local method="$1"
  CMD=(
    "${SCRIPT_DIR}/run_baseline_method.sh" "${method}"
    --top-k "${TOP_K}"
    --timeout "${TIMEOUT}"
    --data-root "${DATA_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --workspace-root "${WORKSPACE_ROOT}"
    --relrag-config "${RELRAG_CONFIG}"
    --log-root "${LOG_ROOT}"
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi
  if [[ "${RETRIEVAL_ONLY}" -eq 1 ]]; then
    CMD+=(--retrieval-only)
  fi
  if [[ "${REBUILD_INDEX}" -eq 1 ]]; then
    CMD+=(--rebuild-index)
  fi
  if [[ "${method}" == "graphrag" && "${GRAPHRAG_PARALLEL_DATASETS}" -eq 1 ]]; then
    CMD+=(--parallel-datasets)
  fi
  if [[ "${method}" == "graphrag" ]]; then
    if [[ -n "${GRAPHRAG_QA_MODE}" ]]; then
      CMD+=(--graphrag_qa_mode "${GRAPHRAG_QA_MODE}")
    fi
    if [[ -n "${COMMUNITY_REPORT_WORKFLOW}" ]]; then
      CMD+=(--community_report_workflow "${COMMUNITY_REPORT_WORKFLOW}")
    fi
  fi

  run_task_logged "baseline_all/${method}" "${LOG_DIR}/baseline_all_${method}.log" "${CMD[@]}"
}

if [[ "${PARALLEL_METHODS}" -eq 1 ]]; then
  DONE=0
  for method in "${METHODS[@]}"; do
    DONE=$((DONE + 1))
    progress_bar "${DONE}" "${TOTAL}" "baseline_all method=${method}"
    run_one_method "${method}"
  done
else
  log_info "parallel_methods=${PARALLEL_METHODS}"
  STARTED=0
  FAIL_COUNT=0
  declare -a RUN_PIDS=()
  declare -a RUN_TAGS=()

  wait_oldest() {
    local pid="${RUN_PIDS[0]}"
    local tag="${RUN_TAGS[0]}"
    local status=0
    set +e
    wait "${pid}"
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      log_info "parallel_fail baseline_all/${tag} exit=${status}"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    else
      log_info "parallel_done baseline_all/${tag}"
    fi
    RUN_PIDS=("${RUN_PIDS[@]:1}")
    RUN_TAGS=("${RUN_TAGS[@]:1}")
  }

  for method in "${METHODS[@]}"; do
    STARTED=$((STARTED + 1))
    progress_bar "${STARTED}" "${TOTAL}" "baseline_all launch method=${method}"
    run_one_method "${method}" &
    RUN_PIDS+=("$!")
    RUN_TAGS+=("${method}")
    if [[ "${#RUN_PIDS[@]}" -ge "${PARALLEL_METHODS}" ]]; then
      wait_oldest
    fi
  done

  while [[ "${#RUN_PIDS[@]}" -gt 0 ]]; do
    wait_oldest
  done

  if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    log_info "all_done baseline_all with_failures=${FAIL_COUNT}"
    exit 1
  fi
fi

log_info "all_done baseline_all"
log_info "summary_log=${SUMMARY_LOG}"
