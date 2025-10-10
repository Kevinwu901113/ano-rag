#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 NQ-dev（nq_dev_rag_raw/）转为 MuSiQue 风格 JSONL：
每条样本：
{
  "id": <example_id>,
  "question": <question>,
  "paragraphs": [{"idx": <cand_idx>, "title": "", "paragraph_text": <text>}, ...]
}
- 仅使用本地已导出的 nq_dev_rag_raw/{corpus.jsonl, queries_dev.jsonl}
- 每条样本限制最多 top_k 段落（含 gold）
"""

import json, argparse, os
from collections import defaultdict

def load_queries(queries_path):
    items = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                items.append(rec)
    return items

def load_corpus_grouped(corpus_path):
    # 将 doc_id = "<example_id>_<cand_idx>" 解析成 (example_id, cand_idx)
    groups = defaultdict(list)
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)
            doc_id = r["doc_id"]
            text = r.get("text", "")
            # 解析 example_id 和 cand_idx
            if "_" in doc_id:
                ex_id, cand_str = doc_id.split("_", 1)
                try:
                    cand_idx = int(cand_str)
                except:
                    # gold 补位等情况，给个大 idx 防止冲突
                    cand_idx = 10**7
                groups[ex_id].append((cand_idx, text))
    # cand_idx 升序
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups

def main(corpus_path, queries_path, out_jsonl, top_k):
    groups = load_corpus_grouped(corpus_path)
    queries = load_queries(queries_path)

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    kept, miss = 0, 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for q in queries:
            ex_id = str(q["query_id"])
            question = q["question"]
            # 候选段落
            cand_list = groups.get(ex_id, [])
            if not cand_list:
                miss += 1
                continue

            # 选前 top_k，并尽量保证 gold 在其中
            gold_id = q.get("gold_doc_id")
            gold_idx = None
            if gold_id and "_" in gold_id:
                try:
                    gold_idx = int(gold_id.split("_", 1)[1])
                except:
                    gold_idx = None

            paras = []
            seen_idx = set()
            # 1) 先放 gold
            if gold_idx is not None:
                for cidx, text in cand_list:
                    if cidx == gold_idx and cidx not in seen_idx and text.strip():
                        paras.append({"idx": cidx, "title": "", "paragraph_text": text})
                        seen_idx.add(cidx)
                        break
            # 2) 再补齐到 top_k
            for cidx, text in cand_list:
                if len(paras) >= top_k:
                    break
                if cidx in seen_idx:
                    continue
                if not text.strip():
                    continue
                paras.append({"idx": cidx, "title": "", "paragraph_text": text})
                seen_idx.add(cidx)

            if not paras:
                miss += 1
                continue

            rec = {
                "id": ex_id,
                "question": question,
                "paragraphs": paras
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] wrote {out_jsonl} ; kept={kept}, miss(no paragraphs)={miss}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="nq_dev_rag_raw/corpus.jsonl")
    ap.add_argument("--queries", default="nq_dev_rag_raw/queries_dev.jsonl")
    ap.add_argument("--out", default="data/nq_dev/musique_style_dev.jsonl")
    ap.add_argument("--top_k", type=int, default=50, help="每条样本最多保留的候选段落数")
    args = ap.parse_args()
    main(args.corpus, args.queries, args.out, args.top_k)
