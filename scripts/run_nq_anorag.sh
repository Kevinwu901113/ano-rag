#!/usr/bin/env bash
set -e

INPUT="data/nq_dev/musique_style_dev.jsonl"
OUT="result/nq_dev/nq_results.anorag.jsonl"
TOPK=50
SUP=4
W=8

mkdir -p "$(dirname "$OUT")"

echo "[RUN] anorag on NQ"
python main_nq.py "$INPUT" "$OUT" \
  --engine anorag \
  --workers $W \
  --topk $TOPK \
  --topk_support $SUP

echo "[CHECK] retrieved_doc_ids non-empty?"
jq -r 'select(.retrieved_doc_ids|length==0)' "$OUT" | head -n 1 || echo "OK: have retrieved_doc_ids"

echo "[EVAL] retrieval"
python eval_retrieval_nq.py --results "$OUT" --qrels nq_dev_rag_raw/qrels_dev.tsv

echo "[EVAL] generation"
python eval_generation_nq.py --results "$OUT" --queries nq_dev_rag_raw/queries_dev.jsonl
