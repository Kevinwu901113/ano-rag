# HotpotQA Experiment Readiness Audit Report

**Date:** 2026-01-16
**Status:** **READY** (with minor action items)

## 1. Readiness Checklist

| Category | Item | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Config** | `--config` support | ✅ OK | `hotpot_entry.py` supports config loading with priority. |
| **Config** | Grid Config | ✅ OK | Created `relrag/config/exp_hotpot_grid.yaml` with full grid. |
| **Data** | Path Resolution | ✅ OK | `ConfigLoader` normalizes paths relative to repo root. |
| **Data** | Output Safety | ✅ OK | `hotpot_entry.py` separates runs by filename (`pred_{split}_{reader}_{mode}.jsonl`). |
| **Retriever** | BM25 | ✅ OK | `BM25IndexBuilder` integrated; builds automatically if missing. |
| **Retriever** | Dense | ✅ OK | `EmbeddingIndexBuilder` integrated; supports caching and FAISS. |
| **Retriever** | Hybrid | ✅ OK | Logic exists to combine BM25 and Dense scores. |
| **Reader** | vLLM | ✅ OK | Configurable via `vllm` section; endpoint check command provided below. |
| **Reader** | OpenAI | ✅ OK | Uses `OPENAI_API_KEY` env var; supports `api_key_env` config. |
| **Metrics** | Calculation | ✅ OK | `score_metrics` computes BLEU-1/4, ROUGE-L, METEOR. |
| **Artifacts** | Output Files | ✅ OK | Generates `pred_*.jsonl` and `summary_*.json`. |

## 2. Recommended Experiment Configuration

Use **`relrag/config/exp_hotpot_grid.yaml`** for all experiments.

**Key Settings:**
- **Retrievers:** `["bm25", "dense", "hybrid"]`
- **Readers:** `["vllm", "openai"]`
- **Top-K:** 10 (configurable per retriever)
- **Data:** `data/hotpot_dev_distractor_500.json` (Default)
- **Output:** `result/hotpot_grid`

## 3. Experiment Commands

### 3.0 Prerequisites & Environment
**Set OpenAI Key:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Check vLLM Connection:**
```bash
# Verify vLLM is running (adjust URL if needed)
curl http://127.0.0.1:8000/v1/models
```
*If connection fails:* Start vLLM with `python -m vllm.entrypoints.openai.api_server ...`

### 3.1 Smoke Test (Quick Verification)
Runs 2 examples for all combinations (BM25/Dense/Hybrid × vLLM/OpenAI).
```bash
python hotpot_entry.py \
  --config relrag/config/exp_hotpot_grid.yaml \
  --output_dir result/hotpot_smoke \
  --limit 2
```

### 3.2 Full Grid Experiment
Runs the full dataset (500 examples or full set if configured) for all combinations.
```bash
python hotpot_entry.py \
  --config relrag/config/exp_hotpot_grid.yaml \
  --output_dir result/hotpot_grid
```

### 3.3 Comparative Experiments
**Retrievers Only (Fix Reader = vLLM):**
```bash
python hotpot_entry.py \
  --config relrag/config/exp_hotpot_grid.yaml \
  --reader vllm \
  --output_dir result/hotpot_retrievers
```

**Models Only (Fix Retriever = Hybrid):**
```bash
python hotpot_entry.py \
  --config relrag/config/exp_hotpot_grid.yaml \
  --retriever hybrid \
  --output_dir result/hotpot_models
```

### 3.4 Base Directory Verification (Non-root Run)
Verify that relative paths in config work from a different directory.
```bash
cd data
python ../hotpot_entry.py \
  --config ../relrag/config/exp_hotpot_grid.yaml \
  --output_dir ../result/test_path_res \
  --limit 2
cd ..
```

## 4. Artifact Verification

**Expected Outputs (in output directory):**
- `pred_dev_vllm_bm25.jsonl`
- `pred_dev_vllm_dense.jsonl`
- `pred_dev_vllm_hybrid.jsonl`
- `pred_dev_openai_bm25.jsonl`
- ... (other combinations)
- `summary_dev.json` (Aggregated metrics)

**Verification Commands:**

**Check Metrics Presence:**
```bash
# Ensure metrics are recorded in the first prediction file found
head -n 1 result/hotpot_smoke/pred_*.jsonl | grep "metrics"
```

**Check Summary Aggregation:**
```bash
# Check if summary contains keys for readers
cat result/hotpot_smoke/summary_dev.json | grep -E '"vllm"| "openai"'
```
