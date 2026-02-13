# Baseline Workspace

This directory contains five standalone baseline methods:
- `lightrag`
- `graphrag`
- `raptor`
- `bm25`
- `dense`

All five methods now use **per-question document pools** from `baseline/data/<dataset>/qa.jsonl` (`row.docs`).
No method retrieves from a cross-question global corpus during runtime.

## Fixed Model Interfaces

- Chat (Qwen / vLLM): `http://127.0.0.1:8000/v1`, model `qwen3-30b-a3b`, `api_key=EMPTY`
- Embedding (vLLM): `http://127.0.0.1:8001/v1`, model `qwen3-embedding`, `api_key=EMPTY`
- Chat (DeepSeek OpenAI mode): `https://api.deepseek.com/v1`, model `deepseek-chat`, key from `OPENAI_API_KEY` (fallback: `DEEPSEEK_API_KEY`)

## Conda Environments (reused)

- `baseline-lightrag`
- `baseline-graphrag`
- `baseline-raptor`

## Core Paths

- Data intermediate layer: `baseline/data/<dataset>/`
- Runners: `baseline/runners/`
- Workspaces/indexes: `baseline/workspaces/`
- Evaluation scripts: `baseline/eval/`
- Outputs: `baseline/results/`

## Quickstart

1. Initialize RAPTOR submodule (once per clone):

```bash
git submodule sync --recursive
git submodule update --init --recursive RAPTOR/raptor
```

2. Check services:

```bash
python baseline/tools/check_services.py
```

3. Build intermediate data (includes `qa.jsonl.docs`):

```bash
python baseline/tools/build_intermediate.py
```

4. Run a single method (example):

```bash
conda run -n baseline-lightrag python baseline/runners/run_bm25_qa.py \
  --dataset hotpotqa --llm_backend qwen --limit 5 --rebuild_index --top_k 5
```

5. Run full smoke matrix (`5 methods x 3 datasets x 2 backends`, `--limit 5`):

```bash
export OPENAI_API_KEY="<deepseek_api_key>"
python baseline/tools/run_smoke_baselines.py --limit 5 --rebuild_index
```

6. Score all available outputs:

```bash
python baseline/eval/score_all.py
```

## Runner CLI (unified)

All runners support:
- `--dataset {hotpotqa,musique,2wiki}`
- `--llm_backend {qwen,deepseek}`
- `--data_root` / `--output_root` / `--workspace_root`
- `--limit`
- `--rebuild_index`
- `--top_k`
- `--answer_max_tokens`
- `--qa_prompt_mode answer_only`

Notes:
- `run_lightrag_qa.py` and `run_graphrag_qa.py` auto-detect embedding dimension when `--embedding_dim <= 0`.
- `run_dense_qa.py` supports optional per-question embedding cache via `--cache_embeddings`.
