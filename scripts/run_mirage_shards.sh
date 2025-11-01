#!/usr/bin/env bash
# Run multiple MIRAGE shards with dynamic vLLM endpoint management.
# Usage:
#   scripts/run_mirage_shards.sh shard_00 shard_01 ...
#
# Configuration knobs (env overrideable):
#   MAX_JOBS   – maximum shards to run concurrently (default: 4)
#   SHARD_DIR  – base directory of split shards
#   RESULT_DIR – root directory for per-shard outputs
#   LOG_DIR    – where to store per-shard logs

set -euo pipefail

MAX_JOBS="${MAX_JOBS:-4}"
SHARD_DIR="${SHARD_DIR:-MIRAGE/mirage/splits}"
RESULT_DIR="${RESULT_DIR:-result}"
LOG_DIR="${LOG_DIR:-logs/mirage_shards}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-logs/vllm}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
EMBEDDING_DEVICE_OVERRIDE="${EMBEDDING_DEVICE_OVERRIDE:-${EMBEDDING_DEVICE:-}}"
STEP_MONITOR_INTERVAL="${STEP_MONITOR_INTERVAL:-8}"
STEP2_BARRIER_TIMEOUT="${STEP2_BARRIER_TIMEOUT:-1800}"
STEP2_BARRIER_POLL="${STEP2_BARRIER_POLL:-5}"
USE_SMALL_FIXTURES="${USE_SMALL_FIXTURES:-0}"
SMALL_FIXTURE_NAME="${SMALL_FIXTURE_NAME:-mirage_small}"
SMALL_DATASET_FILE="${SMALL_DATASET_FILE:-MIRAGE/mirage/dataset_small.json}"
SMALL_DOC_POOL_FILE="${SMALL_DOC_POOL_FILE:-MIRAGE/mirage/doc_pool_small.json}"
FINAL_ANSWER_MODEL="${FINAL_ANSWER_MODEL:-}"

# vLLM launch parameters (aligned with config.yaml defaults)
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VLLM_QUANT="${VLLM_QUANT:-awq_marlin}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.95}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_BATCH_TOKENS="${VLLM_MAX_BATCH_TOKENS:-49152}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-/home/wjk/models/Qwen2.5-7B-AWQ}"

GPU0_PORT="${GPU0_PORT:-8000}"
GPU1_PORT="${GPU1_PORT:-8001}"
VLLM_READINESS_TIMEOUT="${VLLM_READINESS_TIMEOUT:-240}"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*"
}

endpoint_ready() {
  local host="$1"
  local port="$2"
  if curl -s -m 2 "http://${host}:${port}/v1/models" | grep -q '"data"'; then
    return 0
  fi
  return 1
}

wait_for_ready() {
  local host="$1"
  local port="$2"
  local pid="$3"
  local timeout="${VLLM_READINESS_TIMEOUT}"
  local interval=2
  local elapsed=0

  while (( elapsed < timeout )); do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if endpoint_ready "${host}" "${port}"; then
      return 0
    fi
    sleep "${interval}"
    elapsed=$((elapsed + interval))
  done
  return 1
}

declare -A VLLM_PIDS=()
STARTED_ENDPOINTS=()
VLLM_STOPPED=0
MONITOR_PID=""

stop_vllm_instances() {
  local reason="${1:-}"
  if (( VLLM_STOPPED )); then
    return
  fi
  VLLM_STOPPED=1
  local have_pid=0
  for pid in "${VLLM_PIDS[@]}"; do
    if [[ -n "${pid}" ]]; then
      have_pid=1
      break
    fi
  done
  if [[ -n "${reason}" && ${have_pid} -eq 1 ]]; then
    log "${reason}"
  fi
  for name in "${!VLLM_PIDS[@]}"; do
    local pid="${VLLM_PIDS[$name]}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "Stopping vLLM instance '${name}' (pid=${pid})"
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
    VLLM_PIDS["${name}"]=""
  done
  STARTED_ENDPOINTS=()
}

cleanup() {
  stop_vllm_instances "Cleaning up vLLM instances (trap)."
}
trap cleanup EXIT INT TERM

start_vllm_instance() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local host="127.0.0.1"
  local endpoint="http://${host}:${port}/v1"
  local log_file="${VLLM_LOG_DIR}/${name}.log"

  if endpoint_ready "${host}" "${port}"; then
    log "Reusing existing vLLM server '${name}' on ${host}:${port}"
    VLLM_PIDS["${name}"]=""
    STARTED_ENDPOINTS+=("${endpoint}")
    return 0
  fi

  if lsof -Pi ":${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    log "Port ${port} already in use but endpoint not healthy; skipping '${name}'"
    return 1
  fi

  log "Launching vLLM server '${name}' on GPU ${gpu} (port ${port})"
  mkdir -p "${VLLM_LOG_DIR}"

  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  if [[ -n "${VLLM_DOWNLOAD_DIR}" ]]; then
    export HF_HOME="${HF_HOME:-${VLLM_DOWNLOAD_DIR}}"
  fi

  local -a vllm_args=(
    --model "${VLLM_MODEL}"
    --quantization "${VLLM_QUANT}"
    --dtype "${VLLM_DTYPE}"
    --gpu-memory-utilization "${VLLM_GPU_MEM}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --max-num-batched-tokens "${VLLM_MAX_BATCH_TOKENS}"
    --port "${port}"
    --host "${host}"
    --download-dir "${VLLM_DOWNLOAD_DIR}"
  )

  if [[ -n "${VLLM_SERVED_MODEL_NAME}" ]]; then
    vllm_args+=(--served-model-name "${VLLM_SERVED_MODEL_NAME}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python -m vllm.entrypoints.openai.api_server \
    "${vllm_args[@]}" \
    >"${log_file}" 2>&1 &

  local pid=$!
  VLLM_PIDS["${name}"]="${pid}"

  if wait_for_ready "${host}" "${port}" "${pid}"; then
    log "vLLM server '${name}' is ready on ${endpoint}"
    STARTED_ENDPOINTS+=("${endpoint}")
    return 0
  fi

  log "Failed to start vLLM server '${name}' (see ${log_file})"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  fi
  VLLM_PIDS["${name}"]=""
  return 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

RESUME_MODE=0

print_usage() {
  cat <<'USAGE'
Usage: scripts/run_mirage_shards.sh [--resume|--fresh] [--bootstrap-vllm|--internal-vllm] [--embedding-device cpu|cuda|cuda:1] [shard_00 shard_01 ...]

Options:
  --resume   Resume from existing work directories (omit --new)
  --fresh    Force new run (default)
  --bootstrap-vllm
             Pre-launch shared vLLM endpoints (default)
  --internal-vllm
             Skip external vLLM bootstrap and rely on per-process autostart
  --embedding-device <device>
             Override embedding device (exported to ANO_RAG_EMBEDDING_DEVICE)
  --use-small-fixtures
             Use MIRAGE/mirage/*_small.json fixtures instead of shard splits
  --final-answer-model <model>
             Override LM Studio model used for final answer generation (default: openai/gpt-oss-20b)
  --help     Show this help

If no shard names are provided, shard_00 ... shard_03 will be used.
USAGE
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE=1
      shift
      ;;
    --fresh)
      RESUME_MODE=0
      shift
      ;;
    --bootstrap-vllm|--use-external-vllm)
      BOOTSTRAP_VLLM=1
      shift
      ;;
    --internal-vllm|--no-bootstrap-vllm)
      BOOTSTRAP_VLLM=0
      shift
      ;;
    --embedding-device)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      EMBEDDING_DEVICE_OVERRIDE="$2"
      shift 2
      ;;
    --use-small-fixtures|--small-fixtures|--small)
      USE_SMALL_FIXTURES=1
      shift
      ;;
    --final-answer-model)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      FINAL_ANSWER_MODEL="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -eq 0 ]]; then
  if [[ ${USE_SMALL_FIXTURES} -eq 1 ]]; then
    SHARDS=("${SMALL_FIXTURE_NAME}")
  else
    SHARDS=("shard_00" "shard_01" "shard_02" "shard_03")
  fi
else
  SHARDS=("$@")
fi

mkdir -p "${LOG_DIR}"

DEFAULT_FINAL_ANSWER_MODEL="openai/gpt-oss-20b"
if [[ -n "${FINAL_ANSWER_MODEL}" ]]; then
  export ANO_RAG_FINAL_ANSWER_MODEL="${FINAL_ANSWER_MODEL}"
elif [[ -z "${ANO_RAG_FINAL_ANSWER_MODEL:-}" ]]; then
  export ANO_RAG_FINAL_ANSWER_MODEL="${DEFAULT_FINAL_ANSWER_MODEL}"
fi
log "Final answer LM Studio model: ${ANO_RAG_FINAL_ANSWER_MODEL}"

TOTAL_SHARDS=${#SHARDS[@]}
STEP2_BARRIER_DIR=""
if (( TOTAL_SHARDS > 1 )); then
  local_barrier_root="${LOG_DIR}/step2_barrier"
  mkdir -p "${local_barrier_root}"
  STEP2_BARRIER_TOKEN="$(date '+%Y%m%d_%H%M%S')_${RANDOM}_$$"
  STEP2_BARRIER_DIR="${local_barrier_root}/${STEP2_BARRIER_TOKEN}"
  mkdir -p "${STEP2_BARRIER_DIR}"
  log "Step 2 barrier active: ${STEP2_BARRIER_DIR} (expecting ${TOTAL_SHARDS} shards, timeout=${STEP2_BARRIER_TIMEOUT}s)"
fi

# ---------------------------------------------------------------------------
# Optional: bootstrap external vLLM endpoints
# ---------------------------------------------------------------------------

AVAILABLE_ENDPOINTS=()

if [[ ${BOOTSTRAP_VLLM} -eq 1 ]]; then
  STARTED_ENDPOINTS=()
  start_vllm_instance "vllm_gpu0" 0 "${GPU0_PORT}" || true
  start_vllm_instance "vllm_gpu1" 1 "${GPU1_PORT}" || true

  AVAILABLE_ENDPOINTS=("${STARTED_ENDPOINTS[@]}")
  if [[ ${#AVAILABLE_ENDPOINTS[@]} -gt 0 ]]; then
    mapfile -t AVAILABLE_ENDPOINTS < <(printf "%s\n" "${AVAILABLE_ENDPOINTS[@]}" | awk '!seen[$0]++')
  fi

  if [[ ${#AVAILABLE_ENDPOINTS[@]} -eq 0 ]]; then
    stop_vllm_instances
    log "Failed to start any vLLM server; aborting."
    exit 1
  fi

  export ANO_RAG_DISABLE_VLLM_AUTOSTART=1
  export ANO_RAG_VLLM_ENDPOINTS="$(IFS=,; echo "${AVAILABLE_ENDPOINTS[*]}")"
  log "Using shared vLLM endpoints: ${ANO_RAG_VLLM_ENDPOINTS}"
else
  # Ensure environment does not force-disable internal autostart
  unset ANO_RAG_DISABLE_VLLM_AUTOSTART || true
  unset ANO_RAG_VLLM_ENDPOINTS || true
  log "Using internal vLLM autostart (no external servers)."
fi

if [[ -n "${EMBEDDING_DEVICE_OVERRIDE}" ]]; then
  export ANO_RAG_EMBEDDING_DEVICE="${EMBEDDING_DEVICE_OVERRIDE}"
  log "Embedding device override: ${ANO_RAG_EMBEDDING_DEVICE}"
fi

# ---------------------------------------------------------------------------
# Shard execution helpers
# ---------------------------------------------------------------------------

pids=()

run_shard() {
  local shard="$1"
  local shard_path="${SHARD_DIR}/${shard}"
  local dataset doc_pool

  if [[ ${USE_SMALL_FIXTURES} -eq 1 ]]; then
    dataset="${SMALL_DATASET_FILE}"
    doc_pool="${SMALL_DOC_POOL_FILE}"
  else
    dataset="${shard_path}/dataset.json"
    doc_pool="${shard_path}/doc_pool.json"
  fi

  if [[ ! -f "${dataset}" ]]; then
    echo "[${shard}] dataset not found: ${dataset}" >&2
    return 1
  fi
  if [[ ! -f "${doc_pool}" ]]; then
    echo "[${shard}] doc_pool not found: ${doc_pool}" >&2
    return 1
  fi

  local shard_log="${LOG_DIR}/${shard}.log"

  echo "[${shard}] running -> ${shard_log}"
  if [[ ${USE_SMALL_FIXTURES} -eq 1 ]]; then
    local fixture_msg="[${shard}] using MIRAGE small fixtures (dataset=${dataset}, doc_pool=${doc_pool})"
    echo "${fixture_msg}"
    echo "${fixture_msg}" >> "${shard_log}"
  fi

  local -a cmd
  if [[ ${RESUME_MODE} -eq 0 ]]; then
    # Fresh run: let main_mirage.py create a new numeric work directory.
    cmd=(
      python
      main_mirage.py
      "${dataset}"
      "mirage_results.jsonl"
      --doc-pool-file "${doc_pool}"
      --atomic-notes-file "mirage_atomic_notes_recall.jsonl"
      --debug
      --new
    )
  else
    # Resume mode: reuse deterministic work directory per shard.
    local work_dir="${RESULT_DIR}/${shard}"
    local output_file="${work_dir}/mirage_results.jsonl"
    local notes_file="${work_dir}/mirage_atomic_notes_recall.jsonl"
    mkdir -p "${work_dir}"
    echo "[${shard}] resume mode: reusing work dir ${work_dir}"
    cmd=(
      python
      main_mirage.py
      "${dataset}"
      "${output_file}"
      --doc-pool-file "${doc_pool}"
      --atomic-notes-file "${notes_file}"
      --work-dir "${work_dir}"
      --debug
    )
  fi
  (
    export ANO_RAG_SHARD_NAME="${shard}"
    export ANO_RAG_STEP2_BARRIER_DIR="${STEP2_BARRIER_DIR}"
    export ANO_RAG_STEP2_TOTAL="${TOTAL_SHARDS}"
    export ANO_RAG_STEP2_BARRIER_TIMEOUT="${STEP2_BARRIER_TIMEOUT}"
    export ANO_RAG_STEP2_BARRIER_POLL="${STEP2_BARRIER_POLL}"
    set -x
    "${cmd[@]}"
  ) &>"${shard_log}"

  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[${shard}] completed successfully"
  else
    echo "[${shard}] failed with exit code ${rc} (see ${shard_log})" >&2
    if [[ -n "${STEP2_BARRIER_DIR}" ]]; then
      touch "${STEP2_BARRIER_DIR}/ABORT" 2>/dev/null || true
    fi
  fi
  return $rc
}

start_job() {
  local shard="$1"
  run_shard "${shard}" &
  local pid=$!
  pids+=("${pid}")
}

wait_for_slot() {
  while [[ ${#pids[@]} -ge ${MAX_JOBS} ]]; do
    local finished_pid
    if finished_pid=$(wait -n 2>/dev/null); then
      :
    else
      sleep 1
      continue
    fi
    local remaining=()
    for pid in "${pids[@]}"; do
      if [[ "${pid}" != "${finished_pid}" ]]; then
        remaining+=("${pid}")
      fi
    done
    pids=("${remaining[@]}")
  done
}

monitor_step3_and_release() {
  local barrier_dir="${STEP2_BARRIER_DIR}"
  local expected="${#SHARDS[@]}"
  declare -A STEP3_SEEN=()

  while (( BOOTSTRAP_VLLM == 1 )) && (( VLLM_STOPPED == 0 )); do
    if [[ -n "${barrier_dir}" ]]; then
      local abort_file="${barrier_dir}/ABORT"
      if [[ -f "${abort_file}" ]]; then
        stop_vllm_instances "Detected barrier abort flag; stopping shared vLLM servers."
        break
      fi

      local reached
      reached=$(find "${barrier_dir}" -maxdepth 1 -name '*.step3' 2>/dev/null | wc -l | tr -d ' ')
      if (( expected > 0 )) && (( reached >= expected )); then
        stop_vllm_instances "All shards reached Step 3 marker; stopping shared vLLM servers."
        break
      fi
    fi

    if (( expected > 0 )); then
      local fallback_reached=0
      for shard in "${SHARDS[@]}"; do
        if [[ "${STEP3_SEEN[$shard]:-0}" -eq 1 ]]; then
          ((fallback_reached++))
          continue
        fi
        local shard_log="${LOG_DIR}/${shard}.log"
        if [[ -f "${shard_log}" ]] && grep -F -q "Step 3: Creating embeddings" "${shard_log}"; then
          STEP3_SEEN[$shard]=1
          ((fallback_reached++))
          log "[monitor] ${shard} reached Step 3 (log fallback)."
        fi
      done
      if (( fallback_reached >= expected )); then
        stop_vllm_instances "All shard logs indicate Step 3; stopping shared vLLM servers."
        break
      fi
    fi

    local any_running=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_running=1
        break
      fi
    done
    if (( any_running == 0 )); then
      stop_vllm_instances "Shard processes finished; cleaning up vLLM servers."
      break
    fi

    sleep "${STEP_MONITOR_INTERVAL}"
  done
}

# ---------------------------------------------------------------------------
# Execute shards
# ---------------------------------------------------------------------------

for shard in "${SHARDS[@]}"; do
  wait_for_slot
  start_job "${shard}"
done

if [[ ${BOOTSTRAP_VLLM} -eq 1 ]]; then
  monitor_step3_and_release &
  MONITOR_PID=$!
fi

exit_code=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done

if [[ -n "${MONITOR_PID}" ]]; then
  wait "${MONITOR_PID}" 2>/dev/null || true
fi

if [[ $exit_code -eq 0 ]]; then
  echo "All requested shards finished successfully."
else
  echo "Some shards reported errors. Check logs in ${LOG_DIR}." >&2
fi

exit $exit_code
