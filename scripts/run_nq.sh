#!/usr/bin/env bash
set -euo pipefail

INPUT="data/nq_dev/musique_style_dev.jsonl"
OUT="result/nq_dev/nq_results.jsonl"

mkdir -p "$(dirname "$OUT")"

python main_nq.py "$INPUT" "$OUT" --workers 4 --topk 20

python eval_retrieval_nq.py \
  --results "$OUT" \
  --qrels nq_dev_rag_raw/qrels_dev.tsv

python eval_generation_nq.py \
  --results "$OUT" \
  --queries nq_dev_rag_raw/queries_dev.jsonl
