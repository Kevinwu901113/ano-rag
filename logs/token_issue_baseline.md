# Token Issue Baseline (Step 0)

Captured at: 2026-02-10 (before applying Step 1+ config/code changes)

## 1) Current vLLM startup command
Source: `logs/vllm_cmd_gpu0.txt`

```bash
/home/wjk/miniconda3/envs/anorag/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8000 --model cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit --gpu-memory-utilization 0.9 --served-model-name qwen3-30b-a3b --trust-remote-code --max-model-len 8192 --max-num-seqs 24 --max-num-batched-tokens 1024 --swap-space 8 --cpu-offload-gb 8 --kv-cache-dtype fp8
```

Key fact:
- Service-side `max-model-len = 8192`.

## 2) Latest observed 400 error text (raw excerpt)
Primary source available in this workspace:
- `/home/wjk/.codex/history.jsonl` line 830 (user-provided runtime error excerpt)

Raw excerpt:

```text
max_tokens ... is too large: 64 ... maximum context length is 8192 ... request has 8190 input tokens (64 > 8192 - 8190)
```

Key facts parsed from error text:
- `max context = 8192`
- `input tokens = 8190` (very close to 8192)
- requested `max_tokens = 64`

## 3) Step-0 config snapshot (`relrag/config/config.yaml`)

### llm
- `llm.max_context_len: 8192`
- `llm.safety_margin_tokens: 256`

### vllm (current)
- `vllm.endpoint: http://127.0.0.1:8000/v1`
- `vllm.model: qwen3-30b-a3b`
- `vllm.temperature: 0.0`
- `vllm.max_tokens: 256`
- `vllm.max_new_tokens: 256`
- `vllm.concurrency.read_timeout_sec: 20.0`
- `vllm.concurrency.retry_total_cap_sec: 30.0`

### llm_profiles
- `llm_profiles.extract.max_tokens: 256`
- `llm_profiles.generate.max_tokens: 128`

### parsing / answer
- `parsing.max_tokens: 768`
- `answer.max_evidence_tokens: 512`

## 4) Baseline acceptance check
- [x] service max-model-len clearly shows `8192`
- [x] error text clearly shows input tokens near context limit (`8190`)
