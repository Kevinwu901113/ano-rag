#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"
LIMIT="${LIMIT:-}"
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
init_logging "baseline_full" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

log_info "plan baseline_full top_k=${TOP_K} (qa + retrieval_only)"

CMD_QA=(
  "${SCRIPT_DIR}/run_baseline_all.sh"
  --top-k "${TOP_K}"
  --timeout "${TIMEOUT}"
  --parallel-methods "${PARALLEL_METHODS}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --workspace-root "${WORKSPACE_ROOT}"
  --relrag-config "${RELRAG_CONFIG}"
  --log-root "${LOG_ROOT}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD_QA+=(--limit "${LIMIT}")
fi
if [[ "${REBUILD_INDEX}" -eq 1 ]]; then
  CMD_QA+=(--rebuild-index)
fi
if [[ "${GRAPHRAG_PARALLEL_DATASETS}" -eq 1 ]]; then
  CMD_QA+=(--graphrag-parallel-datasets)
fi
if [[ -n "${GRAPHRAG_QA_MODE}" ]]; then
  CMD_QA+=(--graphrag_qa_mode "${GRAPHRAG_QA_MODE}")
fi
if [[ -n "${COMMUNITY_REPORT_WORKFLOW}" ]]; then
  CMD_QA+=(--community_report_workflow "${COMMUNITY_REPORT_WORKFLOW}")
fi

run_task_logged "baseline_full/qa" "${LOG_DIR}/baseline_full_qa.log" "${CMD_QA[@]}"

CMD_RET=(
  "${SCRIPT_DIR}/run_baseline_all.sh"
  --top-k "${TOP_K}"
  --timeout "${TIMEOUT}"
  --parallel-methods "${PARALLEL_METHODS}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --workspace-root "${WORKSPACE_ROOT}"
  --relrag-config "${RELRAG_CONFIG}"
  --log-root "${LOG_ROOT}"
  --retrieval-only
)

if [[ -n "${LIMIT}" ]]; then
  CMD_RET+=(--limit "${LIMIT}")
fi
if [[ "${GRAPHRAG_PARALLEL_DATASETS}" -eq 1 ]]; then
  CMD_RET+=(--graphrag-parallel-datasets)
fi
if [[ -n "${GRAPHRAG_QA_MODE}" ]]; then
  CMD_RET+=(--graphrag_qa_mode "${GRAPHRAG_QA_MODE}")
fi
if [[ -n "${COMMUNITY_REPORT_WORKFLOW}" ]]; then
  CMD_RET+=(--community_report_workflow "${COMMUNITY_REPORT_WORKFLOW}")
fi

run_task_logged "baseline_full/retrieval_only" "${LOG_DIR}/baseline_full_retrieval.log" "${CMD_RET[@]}"

log_info "all_done baseline_full"
log_info "summary_log=${SUMMARY_LOG}"
