#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/wjk/.cache/hf
export HF_HUB_CACHE=/home/wjk/.cache/hf
export TRANSFORMERS_CACHE=/home/wjk/.cache/hf
export http_proxy=http://192.168.192.246:7890
export https_proxy=http://192.168.192.246:7890
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '%s\n' "export HF_ENDPOINT=${HF_ENDPOINT}"
printf '%s\n' "export HF_HOME=${HF_HOME}"
printf '%s\n' "export HF_HUB_CACHE=${HF_HUB_CACHE}"
printf '%s\n' "export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
printf '%s\n' "export http_proxy=${http_proxy}"
printf '%s\n' "export https_proxy=${https_proxy}"
printf '%s\n' "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

python run_vllm_dual_gpu.py \
  --cache-dir /home/wjk/.cache/hf \
  --llm-model cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
  --embed-model Qwen/Qwen3-Embedding-8B \
  --llm-port 8000 \
  --embed-port 8001 \
  --llm-gpu 0 \
  --embed-gpu 1 \
  --llm-gpu-mem 0.85 \
  --llm-max-model-len 4096 \
  --llm-max-num-seqs 1 \
  --llm-max-num-batched-tokens 1024 \
  --llm-swap-space 8 \
  --llm-cpu-offload-gb 8 \
  --embed-max-model-len 8192 \
  --embed-max-num-seqs 1 \
  --embed-max-num-batched-tokens 1024 \
  --hf-endpoint https://hf-mirror.com \
  --http-proxy http://192.168.192.246:7890 \
  --https-proxy http://192.168.192.246:7890
