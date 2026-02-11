#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mode="both"
llm_variant=""
extra_args=()
for arg in "$@"; do
  case "${arg}" in
    --llm)
      if [[ "${mode}" == "emb" ]]; then
        printf '%s\n' "Error: cannot use --llm and --emb together." >&2
        exit 1
      fi
      mode="llm"
      ;;
    --emb)
      if [[ "${mode}" == "llm" ]]; then
        printf '%s\n' "Error: cannot use --llm and --emb together." >&2
        exit 1
      fi
      mode="emb"
      ;;
    --gpt)
      if [[ "${llm_variant}" == "qwen" ]]; then
        printf '%s\n' "Error: cannot use --gpt and --qwen together." >&2
        exit 1
      fi
      llm_variant="gpt"
      ;;
    --qwen)
      if [[ "${llm_variant}" == "gpt" ]]; then
        printf '%s\n' "Error: cannot use --gpt and --qwen together." >&2
        exit 1
      fi
      llm_variant="qwen"
      ;;
    *)
      extra_args+=("${arg}")
      ;;
  esac
done
if [[ -z "${llm_variant}" ]]; then
  llm_variant="gpt"
fi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/wjk/.cache/hf
export HF_HUB_CACHE=/home/wjk/.cache/hf
export TRANSFORMERS_CACHE=/home/wjk/.cache/hf
export http_proxy=http://192.168.192.246:7890
export https_proxy=http://192.168.192.246:7890
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_PYTHON=/home/wjk/miniconda3/envs/anorag/bin/python
VLLM_BIN_DIR=/home/wjk/miniconda3/envs/anorag/bin
export PATH="${VLLM_BIN_DIR}:${PATH}"

# 原 qwen 配置（按要求保留注释，不删除）
# LLM_MODEL=cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
# LLM_SERVED_MODEL_NAME=qwen3-30b-a3b
# LLM_MAX_MODEL_LEN=12288
# LLM_MAX_NUM_SEQS=8
# LLM_MAX_NUM_BATCHED_TOKENS=4096
# LLM_CPU_OFFLOAD_GB=8

QWEN_LLM_MODEL=cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
QWEN_SERVED_MODEL_NAME=qwen3-30b-a3b
GPT_LLM_MODEL=openai/gpt-oss-20b
GPT_SERVED_MODEL_NAME=gpt-oss-20b

if [[ "${llm_variant}" == "qwen" ]]; then
  LLM_MODEL="${QWEN_LLM_MODEL}"
  LLM_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME}"
  LLM_MAX_MODEL_LEN=12288
  LLM_MAX_NUM_SEQS=8
  LLM_MAX_NUM_BATCHED_TOKENS=4096
  LLM_CPU_OFFLOAD_GB=8
else
  LLM_MODEL="${GPT_LLM_MODEL}"
  LLM_SERVED_MODEL_NAME="${GPT_SERVED_MODEL_NAME}"
  LLM_MAX_MODEL_LEN=4096
  LLM_MAX_NUM_SEQS=1
  LLM_MAX_NUM_BATCHED_TOKENS=1024
  LLM_CPU_OFFLOAD_GB=0
fi

printf '%s\n' "export HF_ENDPOINT=${HF_ENDPOINT}"
printf '%s\n' "export HF_HOME=${HF_HOME}"
printf '%s\n' "export HF_HUB_CACHE=${HF_HUB_CACHE}"
printf '%s\n' "export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
printf '%s\n' "export http_proxy=${http_proxy}"
printf '%s\n' "export https_proxy=${https_proxy}"
printf '%s\n' "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
printf '%s\n' "export PATH=${PATH}"
printf '%s\n' "llm_variant=${llm_variant} llm_model=${LLM_MODEL} served_name=${LLM_SERVED_MODEL_NAME}"

args=()
if [[ "${mode}" == "llm" ]]; then
  args+=(--llm-only)
elif [[ "${mode}" == "emb" ]]; then
  args+=(--embed-only)
fi
args+=("${extra_args[@]}")

"${VLLM_PYTHON}" run_vllm_dual_gpu.py \
  --cache-dir /home/wjk/.cache/hf \
  --llm-model "${LLM_MODEL}" \
  --llm-served-model-name "${LLM_SERVED_MODEL_NAME}" \
  --embed-model Qwen/Qwen3-Embedding-8B \
  --llm-port 8000 \
  --embed-port 8001 \
  --llm-gpu 0 \
  --embed-gpu 1 \
  --llm-gpu-mem 0.9 \
  --llm-max-model-len "${LLM_MAX_MODEL_LEN}" \
  --llm-max-num-seqs "${LLM_MAX_NUM_SEQS}" \
  --llm-max-num-batched-tokens "${LLM_MAX_NUM_BATCHED_TOKENS}" \
  --llm-swap-space 8 \
  --llm-cpu-offload-gb "${LLM_CPU_OFFLOAD_GB}" \
  --embed-max-model-len 8192 \
  --embed-max-num-seqs 24 \
  --embed-max-num-batched-tokens 1024 \
  --hf-endpoint https://hf-mirror.com \
  --http-proxy http://192.168.192.246:7890 \
  --https-proxy http://192.168.192.246:7890 \
  "${args[@]}"
