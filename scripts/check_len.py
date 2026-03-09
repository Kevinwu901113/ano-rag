
import json
import os

files = [
    "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15/pred_dev_vllm_dense.jsonl",
    "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15/pred_dev_vllm_bm25.jsonl",
    "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15/pred_dev_vllm_hybrid.jsonl"
]

for fpath in files:
    print(f"Checking {os.path.basename(fpath)}...")
    with open(fpath, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            data = json.loads(line)
            raw = data.get("retrieved_context_raw", [])
            topk = data.get("retrieved_context_topk", [])
            print(f"  Example {i}: raw_len={len(raw)}, topk_len={len(topk)}")
            # Check unique titles
            raw_titles = {x.get("title") for x in raw}
            print(f"  Example {i}: unique_raw_titles={len(raw_titles)}")
