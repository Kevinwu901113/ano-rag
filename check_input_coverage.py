import json, argparse

def run(musique_style_jsonl, qrels_tsv):
    # qrels: qid -> set([doc_id])
    gold = {}
    with open(qrels_tsv) as f:
        for line in f:
            qid, did, *rest = line.strip().split("\t")
            gold.setdefault(qid, set()).add(did)

    total, covered = 0, 0
    with open(musique_style_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["id"])
            paras = r.get("paragraphs", [])
            # 输入 paragraphs 的 doc_id 规范：f"{qid}_{idx}"
            present = {f"{qid}_{p.get('idx')}" for p in paras if "idx" in p}
            if qid in gold:
                total += 1
                if gold[qid] & present:
                    covered += 1
    print(f"Queries with gold present in INPUT paragraphs: {covered}/{total} = {covered/total:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/nq_dev/musique_style_dev.jsonl")
    ap.add_argument("--qrels", default="nq_dev_rag_raw/qrels_dev.tsv")
    args = ap.parse_args()
    run(args.input, args.qrels)