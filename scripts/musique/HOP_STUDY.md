# MuSiQue Hop-Stratified Study (Standard RAG vs RelRAG)

This document defines a reproducible, auditable matrix for:
- `standard` = single-shot Standard RAG (`musique_baseline_entry.py`)
- `relrag` = graph + path walk (`musique_entry.py`)
- optional `relrag_no_walk` ablation (`walk_enabled=false`)

## 1) Prepare hop-labeled fixed subsets

```bash
python scripts/musique/hop_study.py prepare \
  --input musique/data/musique_full_v1.0_test.jsonl \
  --output_dir result/musique/hop_study/data_test \
  --hops 2,3,4 \
  --total_samples 500 \
  --total_sample_strategy proportional \
  --answerable_only \
  --dedupe_id
```

Generated artifacts:
- `result/musique/hop_study/data_test/manifest.json`
- `result/musique/hop_study/data_test/*_hop{2,3,4}.jsonl`
- `result/musique/hop_study/data_test/*_hop{2,3,4}_ids.json`
- `result/musique/hop_study/data_test/hop_counts.csv`
- `manifest.json` will include `total_samples_effective` and per-hop `total_allocation_by_hop`.

Tip:
- `--total_sample_strategy balanced` gives near-uniform hop counts.
- `--total_sample_strategy proportional` preserves source hop ratio.

## 2) Run experiment matrix (method x hop x seed)

Example (3 seeds, with no-walk ablation):

```bash
python scripts/musique/hop_study.py matrix \
  --data_manifest result/musique/hop_study/data_test/manifest.json \
  --run_root result/musique/hop_study/runs_test \
  --methods standard,relrag,relrag_no_walk \
  --hops 2,3,4 \
  --seeds 11,29,47 \
  --reader vllm \
  --standard_retriever dense \
  --relrag_retriever hybrid \
  --top_k 10 \
  --top_k_raw 20 \
  --min_overfetch 2.0 \
  --backfill_max_overfetch 4.0 \
  --backfill_step 1.5 \
  --backfill_rounds 3 \
  --workers 1 \
  --split test \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-30b-a3b
```

Progress behavior:
- Terminal progress logs are enabled by default.
- Long runs print heartbeat every 20s by default.
- You can tune or disable:
  - `--heartbeat_sec 10`
  - `--no_progress`

Matrix files:
- `result/musique/hop_study/runs_test/matrix.json`
- `result/musique/hop_study/runs_test/run_matrix.sh`

Per-run audit files (already produced by entry scripts):
- `predictions.jsonl`
- `retrieved_context_raw.jsonl`
- `retrieved_context_topk.jsonl`
- `run_meta.json`
- `config.resolved.json`

## 3) Analyze by hop + CI + curve plotting

```bash
python scripts/musique/hop_study.py analyze \
  --run_root result/musique/hop_study/runs_test \
  --output_dir result/musique/hop_study/analysis_test \
  --bootstrap_samples 5000 \
  --ci 0.95
```

Main outputs:
- `summary_by_hop.csv`
- `decay_summary.csv`
- `paired_bootstrap.csv`
- `conditional_answer_f1_by_joint_coverage.csv`
- `per_question_audit.jsonl`
- `answer_f1_vs_hop.svg`
- `joint_coverage_vs_hop.svg`
- `summary.md`

## Fairness controls to keep fixed

Use identical settings across methods for:
- reader/model (`--reader`, `--endpoint`, `--model`)
- retrieval budget (`--top_k`, `--top_k_raw`, backfill params)
- generation retries and evidence caps (`--llm_retry_on_empty`, `--llm_retry_max_evidence`)
- support-fact policy (`--include_decomposition_sp` or `--no_decomposition_sp`)
- worker count (`--workers`)

For the ablation:
- `relrag_no_walk` is produced by config patching:
  - `retriever.structured.walk_enabled = false`
  - `retriever.structured.multihop_rescue_enabled = false`
