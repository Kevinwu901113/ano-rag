# Baseline Workspace

This directory contains standalone baseline systems for `LightRAG`, `GraphRAG`, and `RAPTOR`.
The implementation is independent of the main project pipeline and only reuses model endpoints and raw datasets.

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

1. Check services:

```bash
python baseline/tools/check_services.py
```

2. Build intermediate data:

```bash
python baseline/tools/build_intermediate.py
```

3. Run Hotpot baselines (example):

```bash
conda run -n baseline-lightrag python baseline/runners/run_lightrag_qa.py --dataset hotpotqa --llm_backend qwen
conda run -n baseline-raptor python baseline/runners/run_raptor_qa.py --dataset hotpotqa --llm_backend qwen
conda run -n baseline-graphrag python baseline/runners/run_graphrag_qa.py --dataset hotpotqa --llm_backend qwen
```

4. Score all available outputs:

```bash
python baseline/eval/score_all.py
```

## Notes

- Runner optional flags:
  - `--limit`: number of QA examples (default `0` = all)
  - `--max_docs`: number of corpus docs for indexing (default `0` = all)
- For full benchmark runs, keep both defaults (`0`).
