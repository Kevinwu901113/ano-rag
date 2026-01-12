Change Report - AWQ Model Switch (Follow-up)

Fixes for reported startup errors:
- run_vllm_dual_gpu.py: improved cache detection to avoid setting HF_HUB_OFFLINE when only config exists (prevents missing-weight errors on first run).
- run_vllm_dual_gpu.py: added config-based quantization detection; if config shows compressed-tensors, the script skips --quantization awq to avoid vLLM mismatch.
- run_vllm_dual_gpu.py: added quantization_reason in vLLM cmd audit log; ensures proxy and HF endpoint defaults are applied when empty.
- run_vllm_dual_gpu.py: added pre-download step (huggingface_hub snapshot_download) for LLM/embedding when weights are missing, to avoid vLLM EngineCore handshake timeout during long first-time downloads.

Final model names (unchanged):
- LLM: cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
- Embedding: Qwen/Qwen3-Embedding-8B

Final vLLM command (from logs/vllm_cmd_gpu0.txt):
- /home/wjk/miniconda3/envs/anorag/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8000 --model cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit --gpu-memory-utilization 0.45 --served-model-name qwen3-30b-a3b --trust-remote-code --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 --swap-space 8 --cpu-offload-gb 4 --kv-cache-dtype fp8

AWQ compatibility decision:
- This model's cached config reports quantization as "compressed-tensors"; vLLM 0.11.0 throws a validation error if --quantization awq is forced.
- Therefore, the script now omits the quantization argument when config indicates compressed-tensors and lets vLLM auto-detect from config.
- If you want to force a quantization argument anyway, pass --llm-quantization awq explicitly.

Cache/mirror behavior:
- HF_ENDPOINT is https://hf-mirror.com, cache paths are set to --cache-dir (/home/wjk/.cache/hf), and proxies are set to http(s)://192.168.192.246:7890.
- HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are enabled only when actual weight files are present in cache, preventing failed offline loads after a partial download.
