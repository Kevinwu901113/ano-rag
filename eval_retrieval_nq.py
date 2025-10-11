#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
from collections import defaultdict

def load_qrels(qrels_path):
    gold = defaultdict(set)
    with open(qrels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                qid, did = parts
                rel = 1
            else:
                qid, did, rel = parts[:3]
            try:
                if int(rel) > 0:
                    gold[str(qid)].add(str(did))
            except:
                gold[str(qid)].add(str(did))
    return gold

def to_str(x): 
    return "" if x is None else str(x)

def extract_doc_ids(result_obj, qid_str):
    """
    按优先级尝试多种字段，返回候选 doc_id 列表（字符串）：
    1) 'retrieved_doc_ids' 或 'doc_ids'
    2) 'retrieved' / 'retrieved_docs' / 'top_docs'（对象里含 doc_id / id / idx）
    3) 'predicted_support_idxs'（段落 idx） -> 构造成 f"{qid_str}_{idx}"
    4) 'top_candidates' / 'ranked_paragraphs' / 'paragraphs'（对象里含 cand_idx/idx）
    """
    # 1) 直接字符串列表
    for key in ("retrieved_doc_ids", "doc_ids"):
        if key in result_obj and isinstance(result_obj[key], list):
            vals = [to_str(x) for x in result_obj[key] if x is not None]
            if vals:
                return vals

    # 2) 对象列表里取 doc_id / id / idx
    for key in ("retrieved", "retrieved_docs", "top_docs"):
        if key in result_obj and isinstance(result_obj[key], list):
            buf = []
            for obj in result_obj[key]:
                if isinstance(obj, dict):
                    if "doc_id" in obj:
                        buf.append(to_str(obj["doc_id"]))
                    elif "id" in obj and "_" in to_str(obj["id"]):
                        buf.append(to_str(obj["id"]))
                    elif "idx" in obj:
                        buf.append(f"{qid_str}_{int(obj['idx'])}")
            if buf:
                return buf

    # 3) 段落 idx 列表
    if "predicted_support_idxs" in result_obj and isinstance(result_obj["predicted_support_idxs"], list):
        return [f"{qid_str}_{int(idx)}" for idx in result_obj["predicted_support_idxs"] if isinstance(idx, int)]

    # 4) 其他候选字段
    for key in ("top_candidates", "ranked_paragraphs", "paragraphs"):
        if key in result_obj and isinstance(result_obj[key], list):
            buf = []
            for obj in result_obj[key]:
                if isinstance(obj, dict):
                    if "cand_idx" in obj:
                        buf.append(f"{qid_str}_{int(obj['cand_idx'])}")
                    elif "idx" in obj:
                        buf.append(f"{qid_str}_{int(obj['idx'])}")
            if buf:
                return buf

    return []

def eval_recall_mrr(results_path, qrels_path, topk=(1,5,10,20), debug_n=10):
    gold = load_qrels(qrels_path)
    gold_qids = set(gold.keys())

    total_lines = 0
    considered = 0
    hit_at_k = {k:0 for k in topk}
    mrr_sum = 0.0

    skipped_no_qid = 0
    skipped_no_candidates = 0
    skipped_qid_not_in_qrels = 0
    debug_samples = []

    with open(results_path) as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("id") or r.get("query_id") or r.get("qid")
            if qid is None:
                skipped_no_qid += 1
                continue
            qid_str = to_str(qid)

            if qid_str not in gold_qids:
                skipped_qid_not_in_qrels += 1
                continue

            docs = extract_doc_ids(r, qid_str)
            if not docs:
                skipped_no_candidates += 1
                continue

            considered += 1
            # 第一个命中位置
            rank_hit = None
            for rank, d in enumerate(docs, 1):
                if d in gold[qid_str]:
                    rank_hit = rank
                    break

            if rank_hit is not None:
                mrr_sum += 1.0 / rank_hit
                for k in topk:
                    if rank_hit <= k:
                        hit_at_k[k] += 1

            if len(debug_samples) < debug_n:
                debug_samples.append({
                    "qid": qid_str,
                    "first_hit_rank": rank_hit,
                    "top5": docs[:5],
                    "gold": list(gold[qid_str])[:5]
                })

    print(f"Total lines in results: {total_lines}")
    print(f"Considered (with candidates & in qrels): {considered}")
    if considered == 0:
        print(f"Skipped: no_qid={skipped_no_qid}, no_candidates={skipped_no_candidates}, qid_not_in_qrels={skipped_qid_not_in_qrels}")
        print("⚠️ 没有可评测记录。请检查结果里的字段名（如 predicted_support_idxs / retrieved_doc_ids）"
              "以及 qrels 的 qid 是否与结果里的 id 一致。")
        return

    for k in topk:
        print(f"Recall@{k}: {hit_at_k[k]/considered:.3f}")
    print(f"MRR: {mrr_sum/considered:.3f}")

    print("\n[Debug samples]")
    for s in debug_samples:
        print(json.dumps(s, ensure_ascii=False))
    print("\nDone.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--qrels", required=True)
    args = ap.parse_args()
    eval_recall_mrr(args.results, args.qrels)
