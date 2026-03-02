#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <method: bm25|dense|lightrag|graphrag|raptor> [options]" >&2
  exit 2
fi

METHOD="$1"
shift

case "${METHOD}" in
  bm25|dense|lightrag|graphrag|raptor) ;;
  *)
    echo "Unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

default_env_for_method() {
  case "$1" in
    bm25|dense|lightrag) echo "${BASELINE_ENV_LIGHTRAG:-baseline-lightrag}" ;;
    graphrag) echo "${BASELINE_ENV_GRAPHRAG:-baseline-graphrag}" ;;
    raptor) echo "${BASELINE_ENV_RAPTOR:-baseline-raptor}" ;;
    *) return 1 ;;
  esac
}

extract_openai_api_key_from_config() {
  local cfg_path="$1"
  if [[ ! -f "${cfg_path}" ]]; then
    return 0
  fi
  awk '
    BEGIN { in_openai = 0 }
    /^[^[:space:]][^:]*:[[:space:]]*$/ {
      if ($0 ~ /^openai:[[:space:]]*$/) {
        in_openai = 1
        next
      }
      if (in_openai == 1) {
        exit
      }
    }
    in_openai == 1 && $0 ~ /^[[:space:]]+api_key:[[:space:]]*/ {
      line = $0
      sub(/^[[:space:]]+api_key:[[:space:]]*/, "", line)
      gsub(/^["\047]/, "", line)
      gsub(/["\047]$/, "", line)
      print line
      exit
    }
  ' "${cfg_path}"
}

bootstrap_deepseek_env_if_needed() {
  local cfg_path="$1"
  if [[ -n "${OPENAI_API_KEY:-}" || -n "${DEEPSEEK_API_KEY:-}" ]]; then
    return 0
  fi
  local cfg_key=""
  cfg_key="$(extract_openai_api_key_from_config "${cfg_path}")"
  cfg_key="${cfg_key#"${cfg_key%%[![:space:]]*}"}"
  cfg_key="${cfg_key%"${cfg_key##*[![:space:]]}"}"
  if [[ -n "${cfg_key}" ]]; then
    export OPENAI_API_KEY="${cfg_key}"
    log_info "deepseek_key_source=config:${cfg_path}"
  fi
}

ensure_local_no_proxy() {
  local local_list="127.0.0.1,localhost"

  if [[ -z "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${local_list}"
  else
    case ",${NO_PROXY}," in
      *,127.0.0.1,*|*,localhost,*) ;;
      *) export NO_PROXY="${NO_PROXY},${local_list}" ;;
    esac
  fi

  if [[ -z "${no_proxy:-}" ]]; then
    export no_proxy="${local_list}"
  else
    case ",${no_proxy}," in
      *,127.0.0.1,*|*,localhost,*) ;;
      *) export no_proxy="${no_proxy},${local_list}" ;;
    esac
  fi
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASETS=(hotpotqa 2wiki musique)

DATA_ROOT="${DATA_ROOT:-baseline/data/aligned_topk10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-baseline/results/aligned_topk10}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-baseline/workspaces/aligned_topk10}"
RELRAG_CONFIG="${RELRAG_CONFIG:-relrag/config/config.yaml}"
TOP_K="${TOP_K:-10}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
LIMIT="${LIMIT:-}"
RETRIEVAL_ONLY=0
REBUILD_INDEX=0
PARALLEL_DATASETS=0
LOG_ROOT="${LOG_ROOT:-logs/aligned_topk10}"
METHOD_ENV="${METHOD_ENV:-$(default_env_for_method "${METHOD}")}"
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
    --parallel-datasets)
      PARALLEL_DATASETS=1
      shift
      ;;
    --top-k)
      TOP_K="${2:-10}"
      shift 2
      ;;
    --timeout)
      REQUEST_TIMEOUT="${2:-180}"
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
    --conda-env)
      METHOD_ENV="${2:-${METHOD_ENV}}"
      shift 2
      ;;
    --graphrag_qa_mode|--graphrag-qa-mode)
      GRAPHRAG_QA_MODE="${2:-}"
      shift 2
      ;;
    --community_report_workflow|--community-report-workflow)
      COMMUNITY_REPORT_WORKFLOW="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

source "${SCRIPT_DIR}/lib.sh"
init_logging "baseline_${METHOD}" "${ROOT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  log_info "fail conda_not_found"
  echo "conda command not found in PATH" >&2
  exit 1
fi

if ! conda run -n "${METHOD_ENV}" python -c "import sys" >/dev/null 2>&1; then
  log_info "fail invalid_conda_env=${METHOD_ENV}"
  echo "Cannot run python in conda env: ${METHOD_ENV}" >&2
  exit 1
fi

bootstrap_deepseek_env_if_needed "${RELRAG_CONFIG}"
ensure_local_no_proxy
if [[ -z "${OPENAI_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  log_info "fail missing_deepseek_key env_and_config_empty"
  echo "Missing DeepSeek key: set OPENAI_API_KEY/DEEPSEEK_API_KEY or openai.api_key in ${RELRAG_CONFIG}" >&2
  exit 1
fi

log_info "plan method=${METHOD} datasets=${#DATASETS[@]} top_k=${TOP_K} retrieval_only=${RETRIEVAL_ONLY}"
log_info "paths data_root=${DATA_ROOT} output_root=${OUTPUT_ROOT} workspace_root=${WORKSPACE_ROOT}"
log_info "conda_env method=${METHOD} env=${METHOD_ENV}"
log_info "parallel_datasets=${PARALLEL_DATASETS}"
log_info "no_proxy=${NO_PROXY:-${no_proxy:-}}"

run_dataset_pair() {
  local dataset="$1"
  CMD_BASE=(
    conda run -n "${METHOD_ENV}" python "baseline/runners/run_${METHOD}_qa.py"
    --dataset "${dataset}"
    --data_root "${DATA_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --workspace_root "${WORKSPACE_ROOT}"
    --top_k "${TOP_K}"
    --request_timeout "${REQUEST_TIMEOUT}"
    --relrag_config "${RELRAG_CONFIG}"
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD_BASE+=(--limit "${LIMIT}")
  fi
  if [[ "${RETRIEVAL_ONLY}" -eq 1 ]]; then
    CMD_BASE+=(--retrieval_only)
  fi
  if [[ "${REBUILD_INDEX}" -eq 1 ]]; then
    CMD_BASE+=(--rebuild_index)
  fi
  if [[ "${METHOD}" == "graphrag" ]]; then
    if [[ -n "${GRAPHRAG_QA_MODE}" ]]; then
      CMD_BASE+=(--graphrag_qa_mode "${GRAPHRAG_QA_MODE}")
    fi
    if [[ -n "${COMMUNITY_REPORT_WORKFLOW}" ]]; then
      CMD_BASE+=(--community_report_workflow "${COMMUNITY_REPORT_WORKFLOW}")
    fi
  fi

  Q_TAG="baseline/${METHOD}/${dataset}/qwen"
  D_TAG="baseline/${METHOD}/${dataset}/deepseek"

  run_task_logged "${Q_TAG}" "${LOG_DIR}/${METHOD}_${dataset}_qwen.log" "${CMD_BASE[@]}" --llm_backend qwen &
  PID_Q=$!
  run_task_logged "${D_TAG}" "${LOG_DIR}/${METHOD}_${dataset}_deepseek.log" "${CMD_BASE[@]}" --llm_backend deepseek &
  PID_D=$!

  wait_pair "${PID_Q}" "${Q_TAG}" "${PID_D}" "${D_TAG}"
}

TOTAL="${#DATASETS[@]}"

if [[ "${PARALLEL_DATASETS}" -eq 1 ]]; then
  if [[ "${METHOD}" != "graphrag" ]]; then
    log_info "warn parallel_datasets_requested_but_only_recommended_for_graphrag"
  fi
  DONE=0
  declare -a DATASET_PIDS=()
  declare -a DATASET_TAGS=()
  for dataset in "${DATASETS[@]}"; do
    DONE=$((DONE + 1))
    progress_bar "${DONE}" "${TOTAL}" "baseline/${METHOD} dataset=${dataset} (dataset-level parallel)"
    run_dataset_pair "${dataset}" &
    DATASET_PIDS+=("$!")
    DATASET_TAGS+=("${dataset}")
  done

  FAIL_COUNT=0
  for idx in "${!DATASET_PIDS[@]}"; do
    pid="${DATASET_PIDS[$idx]}"
    tag="${DATASET_TAGS[$idx]}"
    status=0
    set +e
    wait "${pid}"
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      log_info "dataset_failed method=${METHOD} dataset=${tag} exit=${status}"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    else
      log_info "dataset_done method=${METHOD} dataset=${tag}"
    fi
  done
  if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    exit 1
  fi
else
  DONE=0
  for dataset in "${DATASETS[@]}"; do
    DONE=$((DONE + 1))
    progress_bar "${DONE}" "${TOTAL}" "baseline/${METHOD} dataset=${dataset} (qwen+deepseek parallel)"
    run_dataset_pair "${dataset}"
  done
fi

log_info "all_done method=${METHOD}"
log_info "summary_log=${SUMMARY_LOG}"
