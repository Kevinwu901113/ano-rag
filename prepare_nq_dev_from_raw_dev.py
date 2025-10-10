#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, gzip, json, glob, argparse
from tqdm import tqdm

def iter_dev_examples(dev_dir):
    files = sorted(glob.glob(os.path.join(dev_dir, "nq-dev-*.jsonl.gz")))
    if not files:
        raise FileNotFoundError(f"No dev shards found under {dev_dir}")
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

def tokens_to_text(tokens, start_token, end_token):
    start = max(0, start_token)
    end = max(start, end_token)
    seg = []
    for i in range(start, min(end, len(tokens))):
        t = tokens[i]
        is_html = t.get("html_token") or t.get("is_html") or False
        tok = t.get("token") or t.get("text")
        if tok and not is_html:
            seg.append(tok)
    return " ".join(seg).strip()

def main(dev_dir, out_dir, limit):
    os.makedirs(out_dir, exist_ok=True)
    corpus_fp = open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8")
    queries_fp = open(os.path.join(out_dir, "queries_dev.jsonl"), "w", encoding="utf-8")
    qrels_fp = open(os.path.join(out_dir, "qrels_dev.tsv"), "w", encoding="utf-8")

    seen_doc_ids = set()
    kept_queries = 0

    for ex in tqdm(iter_dev_examples(dev_dir), desc="parse dev"):
        if limit and kept_queries >= limit:
            break
        ex_id = ex.get("example_id") or ex.get("exampleId") or ex.get("id")
        question = ex.get("question_text") or ex.get("question") or ""
        tokens = ex.get("document_tokens") or []
        cands = ex.get("long_answer_candidates") or []
        annos = ex.get("annotations") or []

        gold_doc_id = None
        short_answers_texts = []
        yes_no = "NONE"

        if annos:
            anno = annos[0]
            la = anno.get("long_answer") or {}
            la_start = la.get("start_token", -1)
            la_end = la.get("end_token", -1)
            la_cand_idx = la.get("candidate_index", -1)
            if la_start is not None and la_end is not None and la_start >= 0 and la_end > la_start:
                gold_doc_id = f"{ex_id}_{la_cand_idx if la_cand_idx is not None and la_cand_idx>=0 else 'gold'}"
            for sa in (anno.get("short_answers") or []):
                s, e = sa.get("start_token", -1), sa.get("end_token", -1)
                if s is not None and e is not None and s >= 0 and e > s:
                    short_answers_texts.append(tokens_to_text(tokens, s, e))
            y = anno.get("yes_no_answer")
            if y in ("YES", "NO"):
                yes_no = y

        queries_fp.write(json.dumps({
            "query_id": str(ex_id),
            "question": question,
            "gold_doc_id": gold_doc_id,
            "short_answers": short_answers_texts,
            "yes_no_answer": yes_no
        }, ensure_ascii=False) + "\n")
        kept_queries += 1

        for idx, cand in enumerate(cands):
            st = cand.get("start_token", -1)
            ed = cand.get("end_token", -1)
            if st is None or ed is None or st < 0 or ed <= st:
                continue
            doc_id = f"{ex_id}_{idx}"
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            text = tokens_to_text(tokens, st, ed)
            corpus_fp.write(json.dumps({
                "doc_id": doc_id,
                "text": text,
                "example_id": ex_id,
                "cand_idx": idx
            }, ensure_ascii=False) + "\n")

        if gold_doc_id is not None:
            qrels_fp.write(f"{ex_id}\t{gold_doc_id}\t1\n")

    corpus_fp.close()
    queries_fp.close()
    qrels_fp.close()
    print(f"[OK] wrote {out_dir}/corpus.jsonl, queries_dev.jsonl, qrels_dev.tsv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_dir", type=str,
                    default=os.path.expanduser("~/.ir_datasets/downloads/natural_questions/v1.0/dev"),
                    help="包含 nq-dev-*.jsonl.gz 的目录")
    ap.add_argument("--out_dir", type=str, default="nq_dev_rag_raw")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0 表示全量 dev）")
    args = ap.parse_args()
    main(args.dev_dir, args.out_dir, args.limit)
