#!/usr/bin/env bash
set -euo pipefail

# Proxy required for the initial model download.
export http_proxy="http://192.168.192.246:7890"
export https_proxy="http://192.168.192.246:7890"

#
# Musique single-GPU orchestration script
# - Start vLLM on GPU0
# - Run musique pipeline (note generation + answering via vLLM)
# - Ensure vLLM is terminated on normal exit or error
#

# =====================
# Config
# =====================
DATASET_NAME="${DATASET_NAME:-musique}"
DATASET_PATH="${DATASET_PATH:-data/${DATASET_NAME}_sample/${DATASET_NAME}.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-result_relrag}"
WORKDIR="${WORKDIR:-}"
RUN_ID="${RUN_ID:-}"
TAG="${TAG:-}"     # e.g. dev200-run1
NEW_RUN=${NEW_RUN:-1}

VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
VLLM_SERVED_MODEL="qwen3-30b-a3b"
VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"
GPU0="${GPU0:-0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
VLLM_DOWNLOAD_DIR="${HOME:-/root}/.cache/huggingface"
QUANTIZATION="${QUANTIZATION:-}"   # e.g. awq
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"

# Pipeline concurrency (producer builds notes, consumer answers)
PRODUCER_WORKERS="${PRODUCER_WORKERS:-8}"
CONSUMER_CONCURRENCY="${CONSUMER_CONCURRENCY:-2}"

# =====================
# Helpers
# =====================
log() { printf "\033[1;34m[%s]\033[0m %s\n" "$(date +'%H:%M:%S')" "$*"; }

wait_http_ok() {
  local url="$1"; local retries="${2:-90}"; local sleep_s="${3:-2}"
  local i=0
  while (( i < retries )); do
    if curl --noproxy '*' -sSf -m 2 "$url" >/dev/null; then return 0; fi
    sleep "$sleep_s"; i=$((i+1))
  done
  return 1
}

wait_port_closed() {
  local host="$1"; local port="$2"; local retries="${3:-90}"; local sleep_s="${4:-2}"
  local i=0
  while (( i < retries )); do
    if ! (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then return 0; fi
    sleep "$sleep_s"; i=$((i+1))
  done
  return 1
}

ensure_workspace() {
  local run_id="${RUN_ID}"
  if [[ -z "$run_id" ]]; then
    run_id="$(date +%Y%m%d_%H%M%S)"
  fi

  if [[ -n "$WORKDIR" ]]; then
    WORK_DIR="$WORKDIR"
  else
    WORK_DIR="$RESULT_ROOT/run_${run_id}/${DATASET_NAME}"
  fi
  mkdir -p "$WORK_DIR/artifacts/logs" "$WORK_DIR/artifacts/answers" "$WORK_DIR/artifacts/pending" "$WORK_DIR/artifacts/notes" "$WORK_DIR/preds"

  VLLM_LOG0="$WORK_DIR/artifacts/logs/vllm_gpu0.log"
  VLLM_PID0="$WORK_DIR/artifacts/vllm_gpu0.pid"
  log "Workspace: $WORK_DIR"
  log "Dataset:   $DATASET_PATH"
  log "Notes out: $WORK_DIR/artifacts/notes/notes.musique.jsonl"
}

start_vllm_single() {
  log "Starting vLLM (Unified Script) on GPU${GPU0}:${VLLM_PORT}"
  
  # Export vars for unified script
  export VLLM_HOST="${VLLM_HOST}"
  export CUDA_VISIBLE_DEVICES="${GPU0}"
  
  nohup bash scripts/llm/start_vllm_qwen3_30b_a3b.sh > "$VLLM_LOG0" 2>&1 &
  echo $! > "$VLLM_PID0"

  log "Waiting for vLLM endpoint ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 120 2 || { log "vLLM endpoint not ready"; exit 1; }
  log "vLLM endpoint is healthy."
  STARTED_VLLM=1
}

stop_vllm_and_wait() {
  if [[ -f "$VLLM_PID0" ]]; then
    local pid; pid="$(cat "$VLLM_PID0" || true)"
    if [[ -n "$pid" ]]; then
      log "Stopping vLLM (pid=${pid})"
      kill "$pid" 2>/dev/null || true
      # Graceful wait then force kill if still running
      sleep 2
      if kill -0 "$pid" 2>/dev/null; then
        log "vLLM still running; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  fi
  wait_port_closed "$VLLM_HOST" "$VLLM_PORT" 90 2 || { log "port ${VLLM_PORT} still busy"; return 1; }
  log "vLLM port closed."
  STARTED_VLLM=0
}

cleanup() {
  if [[ ${STARTED_VLLM:-0} -eq 1 ]]; then
    stop_vllm_and_wait || true
  fi
}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--new] [--tag TAG] [--workdir <path>]

Environment overrides:
  DATASET_NAME, DATASET_PATH, RESULT_ROOT, WORKDIR, RUN_ID, TAG, NEW_RUN
  GPU0, DTYPE, MAX_MODEL_LEN, VLLM_BIN, QUANTIZATION
  PRODUCER_WORKERS, CONSUMER_CONCURRENCY

Example:
  GPU0=0 bash scripts/musique/run_musique.sh --new --tag dev200-run1
USAGE
}

STARTED_VLLM=0
trap cleanup EXIT INT TERM

# =====================
# CLI flags
# =====================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --new)
      NEW_RUN=1; shift ;;
    --tag)
      TAG="$2"; shift 2 ;;
    --workdir|--work-dir)
      WORKDIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      usage; exit 1 ;;
  esac
done

# =====================
# Orchestration
# =====================
ensure_workspace
start_vllm_single

log "Launching Musique pipeline"
python scripts/musique/run.py \
  --dataset-path "$DATASET_PATH" \
  --result-root "$RESULT_ROOT" \
  --workdir "$WORK_DIR" \
  --new \
  --tag "$TAG" \
  --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
  --vllm-model "$VLLM_SERVED_MODEL" \
  --producer-workers "$PRODUCER_WORKERS" \
  --consumer-concurrency "$CONSUMER_CONCURRENCY"

stop_vllm_and_wait
log "Done. Answers (if enabled) under: ${WORK_DIR}/preds"
