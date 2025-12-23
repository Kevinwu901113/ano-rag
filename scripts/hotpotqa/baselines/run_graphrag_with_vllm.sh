#!/usr/bin/env bash
set -euo pipefail

# Proxy required for the initial model download.
export http_proxy="http://192.168.192.246:7890"
export https_proxy="http://192.168.192.246:7890"

# Single-GPU helper: start one vLLM (Qwen3-30B-A3B) for triple extraction + answering.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
VLLM_SERVED_MODEL="qwen3-30b-a3b"
VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.7}"
VLLM_DOWNLOAD_DIR="${HOME:-/root}/.cache/huggingface"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
# 默认给出中等批处理以提高吞吐；可通过显式设置 VLLM_EXTRA_ARGS 覆盖
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-batched-tokens 3072 --max-num-seqs 256}"

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

  echo "Starting vLLM using unified script..."
  # Use VLLM_HOST to control binding (unified script defaults to 0.0.0.0)
  export VLLM_HOST="${VLLM_HOST}"
  
  # Run the unified script in background
  nohup bash scripts/llm/start_vllm_qwen3_30b_a3b.sh > "$VLLM_LOG" 2>&1 &
  echo $! > "$PID_FILE"

  echo "Waiting for vLLM on ${VLLM_HOST}:${VLLM_PORT} ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 120 2 || {
    echo "vLLM not ready; see log ${VLLM_LOG}" >&2
    exit 1
  }
  echo "vLLM ready."
}

start_vllm

python scripts/hotpotqa/baselines/run_graphrag.py \
  --dataset "${DATASET_PATH}" \
  --lm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
  --lm-model "${VLLM_SERVED_MODEL}" \
  --lm-timeout "${LM_TIMEOUT}" \
  --lm-max-tokens "${LM_MAX_TOKENS}" \
  --num-workers "${NUM_WORKERS}" \
  ${WORK_DIR:+--work-dir "$WORK_DIR"} \
  "$@"
