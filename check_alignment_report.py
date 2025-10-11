import json, argparse

def run(results_jsonl, qrels_tsv, n=20):
    qrels_qids = set()
    with open(qrels_tsv) as f:
        for line in f:
            if line.strip():
                qrels_qids.add(line.strip().split("\t", 1)[0])

    total, no_qid, qid_not_in_qrels, no_candidates = 0, 0, 0, 0
    bad_samples = {"no_qid":[], "qid_not_in_qrels":[], "no_candidates":[]}

    def to_str(x): return "" if x is None else str(x)

    def extract_docs(r, qid):
        # 和你的评测脚本一致的提取逻辑（简化版）
        if "retrieved_doc_ids" in r and isinstance(r["retrieved_doc_ids"], list):
            return [to_str(x) for x in r["retrieved_doc_ids"]]
        if "predicted_support_idxs" in r and isinstance(r["predicted_support_idxs"], list):
            return [f"{qid}_{int(i)}" for i in r["predicted_support_idxs"] if isinstance(i, int)]
        return []

    with open(results_jsonl) as f:
        for line in f:
            total += 1
            r = json.loads(line)
            qid = r.get("id") or r.get("query_id") or r.get("qid")
            if qid is None:
                no_qid += 1
                if len(bad_samples["no_qid"])<n: bad_samples["no_qid"].append(r)
                continue
            qid = to_str(qid)
            if qid not in qrels_qids:
                qid_not_in_qrels += 1
                if len(bad_samples["qid_not_in_qrels"])<n: bad_samples["qid_not_in_qrels"].append({"id":qid})
                continue
            docs = extract_docs(r, qid)
            if not docs:
                no_candidates += 1
                if len(bad_samples["no_candidates"])<n: bad_samples["no_candidates"].append({"id":qid, "keys":list(r.keys())})
                continue

    print(f"Total lines: {total}")
    print(f"no_qid: {no_qid}, qid_not_in_qrels: {qid_not_in_qrels}, no_candidates: {no_candidates}")
    print("\nExamples(no_qid):", bad_samples["no_qid"])
    print("\nExamples(qid_not_in_qrels):", bad_samples["qid_not_in_qrels"])
    print("\nExamples(no_candidates):", bad_samples["no_candidates"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="result/24/musique_results.jsonl")
    ap.add_argument("--qrels", default="nq_dev_rag_raw/qrels_dev.tsv")
    args = ap.parse_args()
    run(args.results, args.qrels)