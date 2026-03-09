
import json

files = [
    "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15/pred_dev_vllm_dense.jsonl",
    "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15/pred_dev_vllm_bm25.jsonl"
]

data_dense = []
with open(files[0], 'r') as f:
    for line in f:
        data_dense.append(json.loads(line))

data_bm25 = []
with open(files[1], 'r') as f:
    for line in f:
        data_bm25.append(json.loads(line))

print(f"Loaded {len(data_dense)} dense, {len(data_bm25)} bm25")

for i in range(5):
    d = data_dense[i]
    b = data_bm25[i]
    
    print(f"Example {i}: Dense ID={d['id']}, BM25 ID={b.get('id', b.get('_id'))}")
    
    raw_d = d.get("retrieved_context_raw", [])
    raw_b = b.get("retrieved_context_raw", [])
    
    titles_d = [x.get("title") for x in raw_d]
    titles_b = [x.get("title") for x in raw_b]
    
    print(f"  Dense titles ({len(titles_d)}): {titles_d[:3]}...")
    print(f"  BM25 titles ({len(titles_b)}): {titles_b[:3]}...")
    
    gold_sp_d = d.get("gold_sp", [])
    gold_titles_d = {item[0] for item in gold_sp_d}
    
    gold_sp_b = b.get("gold_sp", [])
    gold_titles_b = {item[0] for item in gold_sp_b}
    
    unique_d = []
    seen = set()
    for t in titles_d:
        if t not in seen:
            unique_d.append(t)
            seen.add(t)
            
    unique_b = []
    seen = set()
    for t in titles_b:
        if t not in seen:
            unique_b.append(t)
            seen.add(t)

    print(f"  Gold Dense: {gold_titles_d}")
    print(f"  Gold BM25: {gold_titles_b}")
    
    # Check hits against OWN gold
    hits_d = sum(1 for t in unique_d if t in gold_titles_d)
    hits_b = sum(1 for t in unique_b if t in gold_titles_b)
    
    print(f"  Hits Dense (vs D-Gold): {hits_d}")
    print(f"  Hits BM25 (vs B-Gold): {hits_b}")
