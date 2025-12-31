#!/usr/bin/env bash
set -euo pipefail

# Proxy is required for the initial model download.
export http_proxy="http://192.168.192.246:7890"
export https_proxy="http://192.168.192.246:7890"

MODEL_ID="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
SERVED_MODEL_NAME="qwen3-30b-a3b"
DOWNLOAD_DIR="${HOME}/.cache/huggingface"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="8000"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
DTYPE="${VLLM_DTYPE:-float16}"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"

detect_tp_size() {
  local count=0
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a devs <<< "${CUDA_VISIBLE_DEVICES}"
    for d in "${devs[@]}"; do
      [[ -n "${d}" ]] && count=$((count + 1))
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    count=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
  if [[ -z "${count}" || "${count}" -lt 1 ]]; then
    count=1
  fi
  echo "${count}"
}

TP_SIZE="$(detect_tp_size)"

exec ${VLLM_BIN} \
  --model "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --quantization gptq \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager
