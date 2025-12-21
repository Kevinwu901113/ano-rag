#!/usr/bin/env bash
set -euo pipefail

# Single-GPU helper: start one vLLM (Qwen2.5-7B) for triple抽取,
# answers go to LM Studio (qwen3-30b-a3b). Customise via env vars below.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.7}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
# 默认给出中等批处理以提高吞吐；可通过显式设置 VLLM_EXTRA_ARGS 覆盖
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-batched-tokens 3072 --max-num-seqs 256}"

LMSTUDIO_ENDPOINT="${LMSTUDIO_ENDPOINT:-http://127.0.0.1:1234/v1}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-qwen/qwen3-30b-a3b}"

DATASET_PATH="${DATASET_PATH:-data/hotpotqa/dataset_distractor_200.json}"
LM_TIMEOUT="${LM_TIMEOUT:-180}"
LM_MAX_TOKENS="${LM_MAX_TOKENS:-512}"
WORK_DIR="${WORK_DIR:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LOG_DIR="${LOG_DIR:-}"
if [[ -z "$LOG_DIR" ]]; then
  if [[ -n "$WORK_DIR" ]]; then
    LOG_DIR="${WORK_DIR}/artifacts/logs"
  else
    LOG_DIR="result_relrag/logs"
  fi
fi
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm_graphrag.log"
PID_FILE="$LOG_DIR/vllm_graphrag.pid"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

resolve_model_path() {
  local candidate="$1"
  if [[ ! -d "$candidate" ]]; then
    echo "$candidate"
    return
  fi

  if [[ -f "$candidate/config.json" ]]; then
    echo "$candidate"
    return
  fi

  local config_path
  config_path=$(find "$candidate" -maxdepth 4 -type f -name "config.json" | head -n 1)
  if [[ -n "$config_path" ]]; then
    echo "$(dirname "$config_path")"
  else
    echo "$candidate"
  fi
}

wait_http_ok() {
  local url="$1" retries="${2:-60}" sleep_s="${3:-2}"
  local i=0
  while (( i < retries )); do
    if curl --noproxy '*' -sSf -m 2 "$url" >/dev/null; then return 0; fi
    sleep "$sleep_s"; i=$((i+1))
  done
  return 1
}

start_vllm() {
  if nc -z "$VLLM_HOST" "$VLLM_PORT" 2>/dev/null; then
    echo "Port ${VLLM_PORT} already in use; aborting." >&2
    exit 1
  fi

  VLLM_MODEL_RESOLVED=$(resolve_model_path "$VLLM_MODEL")

  extra_args=()
  if [[ -n "$VLLM_DOWNLOAD_DIR" ]]; then
    extra_args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
  fi
  if [[ -n "$VLLM_GPU_MEMORY_UTIL" ]]; then
    extra_args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL")
  fi
  if [[ -n "$VLLM_QUANTIZATION" ]]; then
    extra_args+=(--quantization "$VLLM_QUANTIZATION")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL_RESOLVED}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${extra_args[@]}" ${VLLM_EXTRA_ARGS} \
    > "$VLLM_LOG" 2>&1 & echo $! > "$PID_FILE"

  echo "Starting vLLM (${VLLM_MODEL_RESOLVED}) on ${VLLM_HOST}:${VLLM_PORT} ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 90 2 || {
    echo "vLLM not ready; see log ${VLLM_LOG}" >&2
    exit 1
  }
  echo "vLLM ready."
}

start_vllm

python scripts/hotpotqa/baselines/run_graphrag.py \
  --dataset "${DATASET_PATH}" \
  --extract-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
  --extract-model "${VLLM_MODEL_RESOLVED:-$VLLM_MODEL}" \
  --answer-endpoint "${LMSTUDIO_ENDPOINT}" \
  --answer-model "${LMSTUDIO_MODEL}" \
  --lm-timeout "${LM_TIMEOUT}" \
  --lm-max-tokens "${LM_MAX_TOKENS}" \
  --num-workers "${NUM_WORKERS}" \
  ${WORK_DIR:+--work-dir "$WORK_DIR"}
