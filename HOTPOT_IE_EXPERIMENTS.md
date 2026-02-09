# HotpotQA Dev500 IE@K Experiments

## 1) Data Path Requirement
- Input dataset must be HotpotQA dev500 in JSONL format.
- Default path used by scripts: `data/hotpot_dev_distractor_500_jsonl.jsonl`
- Required fields per sample:
  - `_id` (or `id`)
  - `supporting_facts` as `[[title, sent_idx], ...]`

## 2) One Command To Run Full Experiment
```bash
python scripts/hotpotqa/run_ie_experiment_dev500.py \
  --config relrag/config/config.yaml \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --output_root result/hotpot_ie_dev500 \
  --reader vllm
```

Default methods:
- `standard_rag` (default `bm25`, configurable by `--standard_retriever`)
- `relrag_pred_on`
- `relrag_wo_predicate`
- Optional: add `--include_random_predicate` for `relrag_random_predicate`

This runner uses `--retrieval_only true` and exports ranked retrieval outputs for offline IE evaluation.
It also passes `--disable_retriever_llm true` to avoid LLM-based retriever stages during retrieval-only runs.

## 3) Evaluate Only (Reuse Existing run_dir)
Single method:
```bash
python scripts/hotpotqa/eval_ie_curve.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --run_dir result/hotpot_ie_dev500/relrag_pred_on \
  --method relrag_pred_on \
  --audit_samples 20
```

All methods without rerun:
```bash
python scripts/hotpotqa/run_ie_experiment_dev500.py \
  --data data/hotpot_dev_distractor_500_jsonl.jsonl \
  --output_root result/hotpot_ie_dev500 \
  --skip_run
```

## 4) Output Directory Structure
```text
result/hotpot_ie_dev500/
  standard_rag/
    config.resolved.json
    run_meta.json
    metrics.json
    predictions.jsonl
    retrieval/
      final_top50.jsonl
    ie_curve.csv
    ie_curve.json
    ie_curve.png
  relrag_pred_on/
    ...
  relrag_wo_predicate/
    ...
  summary/
    ie_curve_all.csv
    ie_curve_all.json
    ie_curve_all.png
```

Each method run stores auditable artifacts:
- `config.resolved.json` (config snapshot)
- `run_meta.json` (run metadata, including predicate mode, overfetch, dedup policy, token budgets)
- `retrieval/final_top50.jsonl` (final ranked retrieval list)
- `title_audit_samples.json` (optional title-system audit, controlled by `--audit_samples`)

## 5) Metric Definition
Let `gold_titles` be the set of titles in `supporting_facts`.

- Effective evidence (rank `i`):
  - `effective_i = 1` if `retrieved_i.doc_title in gold_titles`
  - else `effective_i = 0`

- IE@K:
  - `IE@K = (1/K) * sum_{i=1..K} effective_i`

- Noise@K:
  - `Noise@K = 1 - IE@K`

- SNR@K (optional):
  - `SNR@K = IE@K / (1 - IE@K + 1e-9)`

Padding rule:
- If final ranked list length is `< K`, missing positions are treated as ineffective (`0`).

Title matching policy:
- Prefer exact title match after conservative normalization (strip + whitespace collapse).
- Fallback uses case-insensitive comparison only.
- No fuzzy matching beyond that.
