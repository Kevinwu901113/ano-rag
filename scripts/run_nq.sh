#!/usr/bin/env bash
set -e

INPUT="data/nq_dev/musique_style_dev.jsonl"
OUT="result/nq_dev/nq_results.jsonl"

mkdir -p result/nq_dev

python main_nq.py "$INPUT" "$OUT" --workers 8 --topk 20 --topk_support 3

# 检索评测
python eval_retrieval_nq.py \
  --results "$OUT" \
  --qrels nq_dev_rag_raw/qrels_dev.tsv

# 生成评测
python eval_generation_nq.py \
  --results "$OUT" \
  --queries nq_dev_rag_raw/queries_dev.jsonl

echo "# 结果是否包含检索候选"
jq -r 'select(.retrieved_doc_ids|length==0)' "$OUT" | head -n 1 || echo "OK: have retrieved_doc_ids"

echo "# 是否有支持段落/非空答案"
jq -r 'select(.predicted_support_idxs|length==0 and .predicted_answer=="")' "$OUT" | head -n 1 || echo "OK: support or answer non-empty"

# 整体统计日志提醒
echo "# 请检查日志中的 [STAT] total=... 行"
