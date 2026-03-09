
import json
import os
import argparse
import numpy as np
from typing import List, Set, Dict, Any

def get_unique_retrieved_titles(retrieved_context: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    unique_titles = []
    for item in retrieved_context:
        title = item.get("title")
        if title and title not in seen:
            unique_titles.append(title)
            seen.add(title)
    return unique_titles

def calculate_ie_series(pred_file: str, max_k: int = 50) -> List[float]:
    print(f"Processing {pred_file}...")
    ie_sums = np.zeros(max_k)
    count = 0
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            gold_sp = data.get("gold_sp", [])
            # Extract gold titles. MuSiQue gold_sp is [[title, idx], ...]
            gold_titles = {item[0] for item in gold_sp}
            
            if not gold_titles:
                continue
                
            retrieved_context = data.get("retrieved_context_raw", [])
            if not retrieved_context:
                 retrieved_context = data.get("retrieved_context_topk", [])
            
            unique_retrieved_titles = get_unique_retrieved_titles(retrieved_context)
            
            # Calculate IE@k for k=1 to max_k
            for k in range(1, max_k + 1):
                # Take top k unique retrieved titles
                current_k_titles = unique_retrieved_titles[:k]
                if not current_k_titles:
                    ie = 0.0
                else:
                    # Count hits
                    hits = sum(1 for t in current_k_titles if t in gold_titles)
                    ie = hits / k
                
                ie_sums[k-1] += ie
            
            count += 1
            
    if count == 0:
        return []
        
    return (ie_sums / count).tolist()

def main():
    base_dir = "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15"
    files = {
        "Dense": "pred_dev_vllm_dense.jsonl",
        "BM25": "pred_dev_vllm_bm25.jsonl",
        "Hybrid": "pred_dev_vllm_hybrid.jsonl"
    }
    
    results = {}
    
    for name, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            results[name] = calculate_ie_series(file_path, max_k=50)
        else:
            print(f"Warning: {file_path} not found.")
            
    # Print results in a format easy to copy
    print("\nIE@K Results (1-50):")
    print("K\tDense\tBM25\tHybrid")
    
    # Check if we have results
    if not results:
        print("No results found.")
        return

    # Assume all have same length (50)
    for k in range(1, 51):
        dense_val = results.get("Dense", [0]*50)[k-1]
        bm25_val = results.get("BM25", [0]*50)[k-1]
        hybrid_val = results.get("Hybrid", [0]*50)[k-1]
        print(f"{k}\t{dense_val:.4f}\t{bm25_val:.4f}\t{hybrid_val:.4f}")

if __name__ == "__main__":
    main()
