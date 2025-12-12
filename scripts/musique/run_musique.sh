#!/usr/bin/env bash
set -euo pipefail

#
# Musique single-GPU orchestration script
# - Start vLLM on GPU0
# - Run musique pipeline (note generation + LM Studio answering)
# - Ensure vLLM is terminated on normal exit or error
#

# =====================
# Config (env overridable)
# =====================
DATASET_NAME="${DATASET_NAME:-musique}"
DATASET_PATH="${DATASET_PATH:-data/${DATASET_NAME}_sample/${DATASET_NAME}.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-result}"
TAG="${TAG:-}"     # e.g. dev200-run1
NEW_RUN=${NEW_RUN:-1}

VLLM_MODEL="${VLLM_MODEL:-qwen2.5-7b-instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU0="${GPU0:-0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
QUANTIZATION="${QUANTIZATION:-}"   # e.g. awq
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"

# LM Studio (optional, if provided answers will be generated)
LMSTUDIO_ENDPOINT="${LMSTUDIO_ENDPOINT:-}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-}"

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
  mkdir -p "$RESULT_ROOT"
  # list existing workspaces: NNN-<dataset>[-tag]
  local entries; entries=$(ls -1 "$RESULT_ROOT" 2>/dev/null | grep -E "^[0-9]{3}-${DATASET_NAME}($|-.*)" || true)
  local next_idx="000"
  if [[ "${NEW_RUN}" -eq 1 || -z "$entries" ]]; then
    if [[ -n "$entries" ]]; then
      local last; last=$(echo "$entries" | sort | tail -n 1)
      local last_prefix; last_prefix=${last%%-*}
      local next_val=$((10#$last_prefix + 1))
      next_idx=$(printf "%03d" "$next_val")
    else
      next_idx="000"
    fi
  else
    local last; last=$(echo "$entries" | sort | tail -n 1)
    next_idx=${last%%-*}
  fi
  local name_suffix="${DATASET_NAME}"
  if [[ -n "$TAG" ]]; then name_suffix="${name_suffix}-${TAG}"; fi
  WORK_DIR="$RESULT_ROOT/${next_idx}-${name_suffix}"
  mkdir -p "$WORK_DIR" "$WORK_DIR/logs" "$WORK_DIR/answers" "$WORK_DIR/pending"

  VLLM_LOG0="$WORK_DIR/logs/vllm_gpu0.log"
  VLLM_PID0="$WORK_DIR/vllm_gpu0.pid"
  log "Workspace: $WORK_DIR"
  log "Dataset:   $DATASET_PATH"
  log "Notes out: $WORK_DIR/notes/notes.musique.jsonl"
}

start_vllm_single() {
  log "Starting vLLM on GPU${GPU0}:${VLLM_PORT} (model=${VLLM_MODEL})"
  local extra_args=()
  if [[ -n "$VLLM_DOWNLOAD_DIR" ]]; then
    extra_args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
  fi
  if [[ -n "$QUANTIZATION" ]]; then
    extra_args+=(--quantization "$QUANTIZATION")
  fi
  if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    extra_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
  fi

  CUDA_VISIBLE_DEVICES="${GPU0}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${extra_args[@]}" \
    > "$VLLM_LOG0" 2>&1 & echo $! > "$VLLM_PID0"

  log "Waiting for vLLM endpoint ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 90 2 || { log "vLLM endpoint not ready"; exit 1; }
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
  $(basename "$0") [--new] [--tag TAG]

Environment overrides:
  DATASET_NAME, DATASET_PATH, RESULT_ROOT, TAG, NEW_RUN
  VLLM_MODEL, VLLM_HOST, VLLM_PORT, GPU0, DTYPE, MAX_MODEL_LEN, VLLM_BIN, VLLM_DOWNLOAD_DIR, QUANTIZATION
  LMSTUDIO_ENDPOINT, LMSTUDIO_MODEL, PRODUCER_WORKERS, CONSUMER_CONCURRENCY

Example:
  VLLM_DOWNLOAD_DIR=/home/user/models VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
  QUANTIZATION=awq bash scripts/musique/run_musique.sh --new --tag dev200-run1 \
    LMSTUDIO_ENDPOINT=http://127.0.0.1:1234/v1 LMSTUDIO_MODEL=Qwen2.5-7B-Instruct
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
  --work-dir "$WORK_DIR" \
  --new \
  --tag "$TAG" \
  --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
  --vllm-model "$VLLM_MODEL" \
  ${LMSTUDIO_ENDPOINT:+--lmstudio-endpoint "$LMSTUDIO_ENDPOINT"} \
  ${LMSTUDIO_MODEL:+--lmstudio-model "$LMSTUDIO_MODEL"} \
  --producer-workers "$PRODUCER_WORKERS" \
  --consumer-concurrency "$CONSUMER_CONCURRENCY"

stop_vllm_and_wait
log "Done. Answers (if enabled) under: ${WORK_DIR}/answers"
