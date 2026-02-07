# UltraDomain LightRAG-Style Experiment Protocol (v1)

This protocol freezes the experimental setup for UltraDomain Mix + Legal replication. Any change requires a new protocol version file (e.g. `experiment_protocol_v2.md`) and updating `result/ultradomain/lightrag_protocol_v1_mix_legal/run_meta/protocol_version`.

## 0) Scope
- Domains: **Mix** and **Legal** only.
- Output root: `result/ultradomain/lightrag_protocol_v1_mix_legal/`
- All runs must follow this protocol exactly.

## 1) Dataset Selection (MVP)
- Selected domains: **Mix + Legal**.
- Rationale: Mix is smaller and Legal is larger, creating a scale gradient. This makes it easy to observe whether structured retrieval benefits persist as corpus size increases. Legal also tends to be more structurally dense, providing a second axis of difference beyond size.

## 2) UltraDomain Data Schema (HF)
- Use HuggingFace UltraDomain dataset.
- Loader policy (frozen):
  - First try `datasets.load_dataset`.
  - If it fails (e.g., schema mismatch or network issues), fallback to direct JSON/JSONL reading via `huggingface_hub`.
  - If Hub listing is unavailable, fallback again to latest local snapshot under HF cache.
- Domain mapping:
  - Normalize from first available candidate in order: `label`, `dataset`, `domain`, `__domain_hint`.
  - `__domain_hint` is injected from source filename stem when fallback loader reads raw files.
- Fields used:
  - `context` as document text.
  - `context_id` or `_id` as `doc_id` (fallback: `{domain}_{row_index}`).
  - `meta.title` as document title (if present).
- Document record stored as:
  - `doc_id`, `title`, `text`, `dataset`, `meta`.

## 3) Chunking (Frozen)
- Tokenizer (local-first):
  - Prefer local tokenizer path from `ultradomain.tokenizer.local_path` or env `ULTRADOMAIN_TOKENIZER_PATH`.
  - Preferred model id: `qwen3-30b-a3b` (via `${vllm.model}`).
  - Fallback model id: `deepseek-ai/DeepSeek-V3.2`.
  - Default behavior uses `local_files_only=true` to avoid online dependency.
- Sentence-aware chunking (sentence as atomic unit):
  - Split text into sentences using **`TextUtils.split_with_spans`** (rule-based).
  - Compute per-sentence token length using DeepSeek tokenizer.
  - Pack sentences in order to target `chunk_size = 1200 tokens`.
- Overlap:
  - Target `overlap_tokens ≈ 100` using sentence backtracking: after emitting a chunk [i, j), backtrack whole sentences from j-1 until overlap tokens ≥ 100, then start next chunk at that sentence.
- Tail handling:
  - **Keep the final tail chunk** even if <1200 tokens (do not drop remaining sentences).
- Chunk ID: `"{doc_id}::t{start}_{end}"` where `start/end` are token offsets.
- Validation:
  - Max chunk length <= 1200 tokens (except rare single-sentence overflow).
  - Fewer than 1% chunks with token_count < 200 (tail chunks may contribute).
  - `chunk_id` is stable and idempotent across runs.

## 4) Question Generation (Frozen, 125 per domain)
- Target: **125 questions per domain** via the LightRAG Persona→Task→Question procedure.
- Process:
  - Generate 5 users.
  - For each user, generate 5 tasks.
  - For each (user, task), generate 5 questions.
- Prompt (JSON output):
  - System: 你是一个评测集生成器。你的任务是为给定语料库生成高层次“sensemaking”问题集合。
  - User: 给你一段“数据集描述 + 语料库摘要（可选）”。请生成：
    - 五个 RAG 用户（每个含：name/背景/专长/动机/提问风格）
    - 每个用户五个任务（每个任务含：目标/约束/关心点）
    - 每个（用户，任务）生成五个问题：必须需要跨文档/跨段落综合，避免单段 factoid；问题应可用语料回答但需要全局理解。
    - 输出严格 JSON：users[5]{profile,tasks[5]{task_desc,questions[5]}}。
- Generation params (frozen): `temperature=0.7`, `top_p=1.0`, `max_output_tokens=4096`.
- Quality gate:
  - LLM validator labels each question as `multi-doc` or `single-factoid`.
  - If pass rate < 95%, regenerate failed questions once.

## 5) Answer LLM (Frozen)
- Provider: OpenAI-compatible local service.
- Base URL: `http://127.0.0.1:8000/v1`.
- Model: `qwen3-30b-a3b`.
- API key env: `OPENAI_API_KEY` (set to `sk-no-key-required` when keyless service is used).
- Answer params: `temperature=0.2`, `top_p=1.0`, `max_output_tokens=1024`.
- Context length: 128K.
- Same model for **all systems**.

## 6) Judge LLM (Frozen)
- Model: `qwen3-30b-a3b` (same as answer LLM).
- Base URL: `http://127.0.0.1:8000/v1`.
- Params: `temperature=0`, `top_p=1.0`.
- Dimensions: Comprehensiveness, Diversity, Empowerment, Overall.
- A/B bias control: alternate order by question_id parity.
- Tie handling: **0.5** to each side.

## 7) Systems (MVP 4)
- RelRAG-full: structured + hybrid, **reranker enabled**.
- BM25-only: chunk BM25 only, reranker disabled.
- Dense-only: chunk FAISS only, reranker disabled.
- Hybrid-only: chunk BM25 + dense fusion, reranker disabled.

## 8) Retrieval Budget Alignment (Frozen)
- Prefilter: top-k=40 per retrieval channel.
- Final context selection: **12,000 tokens** (DeepSeek tokenizer), concatenated in rank order, no chunk splitting.
- Token budget applies to retrieved evidence only (system prompt not counted).

## 9) Outputs (Frozen)
All outputs live under:
`result/ultradomain/lightrag_protocol_v1_mix_legal/`

Required files:
- `chunks/chunks_mix.jsonl`, `chunks/chunks_legal.jsonl`
- `chunks/chunk_stats.json`
- `questions/questions_mix.json`, `questions/questions_legal.json`
- `questions/questions_manifest.json`
- `answers/{dataset}/{system}.jsonl`
- `judge/{dataset}/{pair}.jsonl`
- `summary/summary_winrate.md`
- `run_meta/system_configs.json`, `run_meta/run_config.json`

## 10) Acceptance Criteria
- This protocol is frozen for v1. Any modification requires a new protocol file and version update.
- Chunking and question generation must pass their validation rules.
- All questions must have answers for all 4 systems.
- Judge outputs must be valid JSON lines with all required fields.
