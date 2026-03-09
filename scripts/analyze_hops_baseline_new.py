import json
import re
import os
from collections import defaultdict

base_path = "/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154/pred"

def get_hop_scores(file_path):
    hop_scores = defaultdict(list)
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return {}
        
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                sample_id = data.get('id', '')
                # Try to get F1 from metrics
                metrics = data.get('metrics', {})
                f1 = metrics.get('f1', 0.0)
                
                hop_match = re.match(r'^(\d+)hop', sample_id)
                if hop_match:
                    hop = int(hop_match.group(1))
                    hop_scores[hop].append(f1)
            except json.JSONDecodeError:
                continue
    
    # Calculate average F1 per hop for this file
    avg_hop_scores = {}
    for hop, scores in hop_scores.items():
        if scores:
            avg_hop_scores[hop] = sum(scores) / len(scores)
    return avg_hop_scores

# 1. Process BM25
bm25_qwen = get_hop_scores(os.path.join(base_path, "bm25/musique/qwen/pred.jsonl"))
bm25_deepseek = get_hop_scores(os.path.join(base_path, "bm25/musique/deepseek/pred.jsonl"))

# 2. Process Dense
dense_qwen = get_hop_scores(os.path.join(base_path, "dense/musique/qwen/pred.jsonl"))
dense_deepseek = get_hop_scores(os.path.join(base_path, "dense/musique/deepseek/pred.jsonl"))

print("BM25 Qwen:", bm25_qwen)
print("BM25 DeepSeek:", bm25_deepseek)
print("Dense Qwen:", dense_qwen)
print("Dense DeepSeek:", dense_deepseek)

# 3. Find Best per Hop for BM25 and Dense
hops = [2, 3, 4]
best_bm25 = {}
best_dense = {}

for hop in hops:
    # BM25 Best
    s1 = bm25_qwen.get(hop, 0.0)
    s2 = bm25_deepseek.get(hop, 0.0)
    best_bm25[hop] = max(s1, s2)
    
    # Dense Best
    d1 = dense_qwen.get(hop, 0.0)
    d2 = dense_deepseek.get(hop, 0.0)
    best_dense[hop] = max(d1, d2)

print("\nBest BM25 per hop:", best_bm25)
print("Best Dense per hop:", best_dense)

# 4. Average
final_scores = {}
for hop in hops:
    final_scores[hop] = (best_bm25[hop] + best_dense[hop]) / 2

print("\nFinal Average Scores (Standard RAG):")
for hop in hops:
    print(f"Hop {hop}: {final_scores[hop]:.4f}")
