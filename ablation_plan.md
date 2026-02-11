# HotpotQA Sentence Split vs Fixed Chunking Ablation Study Plan

## Status: Implemented

## 1. Objective
Compare the performance of "Sentence-aware Splitting" (current method) against "Traditional Fixed-size Chunking" (baseline) on the HotpotQA-500 dataset.

## 2. Architecture & Modules

### 2.1 Chunking Strategy Abstraction
*   **Module**: `relrag/doc/chunking_strategies.py`
*   **Interface**: `Chunker`
*   **Implementations**:
    *   `SentenceAwareChunker`: Replicates existing logic (1 chunk per doc for HotpotQA, respecting sentence boundaries).
    *   `FixedWindowChunker`: Sliding window (default 256 tokens, 32 overlap).

### 2.2 Pipeline Modification (`hotpot_entry.py`)
*   Added arguments: `--chunking_method`, `--chunking_size`, `--chunking_overlap`.
*   Integrated `Chunker` into `_process_example` and `_ensure_index`.

### 2.3 Experiment Runner (`scripts/hotpotqa/run_ablation_study.py`)
*   **5-Fold CV**: Splits data into 5 folds.
*   **Execution**: Runs `sentence` and `fixed` methods on each fold.
*   **Config**: Uses `vllm` reader and `hybrid` retriever (LLM reranker disabled for stability/speed).

### 2.4 Advanced Evaluation (`scripts/hotpotqa/eval_ablation.py`)
*   **Metrics**: Recall@K, NDCG@K (with rank decay), IE@K.
*   **Statistics**: Bootstrap 95% CI, Paired t-test.
*   **Output**: `report.txt`, `table.tex`, `recall_curve.png`, `ndcg_curve.png`.

## 3. Usage

```bash
# Run full 5-fold experiment
python scripts/hotpotqa/run_ablation_study.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --output_root result/ablation_study \
  --folds 5 \
  --workers 4
```

## 4. Verification
*   Unit tests: `tests/test_chunking_strategies.py`
*   Smoke test passed (end-to-end pipeline functional).
