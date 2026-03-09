
import json
import os
import numpy as np
from typing import List, Dict, Set, Any

def get_unique_retrieved_titles(items: List[Dict[str, Any]]) -> List[str]:
    """
    Deduplicates items by title, keeping the original order (first occurrence).
    items: list of dicts, each must have a "title" field.
    """
    seen = set()
    unique_titles = []
    for item in items:
        title = item.get("title")
        if title and title not in seen:
            unique_titles.append(title)
            seen.add(title)
    return unique_titles

def load_gold_map(gold_source_file: str) -> Dict[str, Set[str]]:
    """
    Loads {id: {gold_title1, gold_title2, ...}} from a file containing 'gold_sp'.
    """
    print(f"Loading gold map from {gold_source_file}...")
    gold_map = {}
    with open(gold_source_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data.get("id")
                gold_sp = data.get("gold_sp", [])
                # gold_sp is usually [[title, idx], ...]
                titles = {item[0] for item in gold_sp}
                if qid:
                    gold_map[qid] = titles
            except json.JSONDecodeError:
                continue
    print(f"Loaded gold map for {len(gold_map)} queries.")
    return gold_map

def calculate_ie_series_generic(file_path: str, gold_map: Dict[str, Set[str]], 
                              list_field: str = "retrieved_context_raw",
                              max_k: int = 50) -> List[float]:
    print(f"Processing {os.path.basename(file_path)}...")
    ie_sums = np.zeros(max_k)
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            qid = data.get("id")
            if qid not in gold_map:
                # If ID mismatch or missing in gold source, skip or handle?
                # For this task, we assume the gold source covers the dev set.
                continue
                
            gold_titles = gold_map[qid]
            if not gold_titles:
                continue
            
            # Extract retrieved items
            retrieved_items = data.get(list_field, [])
            if not retrieved_items:
                # Fallback or alternative field check
                if list_field == "retrieved_context_raw":
                    retrieved_items = data.get("retrieved_context_topk", [])
            
            unique_titles = get_unique_retrieved_titles(retrieved_items)
            
            # Calculate IE@k
            for k in range(1, max_k + 1):
                current_k = unique_titles[:k]
                if not current_k:
                    ie = 0.0
                else:
                    hits = sum(1 for t in current_k if t in gold_titles)
                    ie = hits / k
                ie_sums[k-1] += ie
                
            count += 1
            
    if count == 0:
        print(f"Warning: No valid entries processed for {file_path}")
        return [0.0] * max_k
        
    return (ie_sums / count).tolist()

def main():
    base_exp15 = "/home/wjk/workplace/nq/ano-rag/result/musique_experiment_15"
    base_baseline = "/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154"
    
    # 1. Load Gold Map from Exp 15 Dense file (most reliable source here)
    gold_source = os.path.join(base_exp15, "pred_dev_vllm_dense.jsonl")
    gold_map = load_gold_map(gold_source)
    
    # 2. Define configurations
    # Format: (Label, FilePath, ListFieldName)
    configs = [
        ("Exp15_Dense", os.path.join(base_exp15, "pred_dev_vllm_dense.jsonl"), "retrieved_context_raw"),
        ("Exp15_BM25", os.path.join(base_exp15, "pred_dev_vllm_bm25.jsonl"), "retrieved_context_raw"),
        ("Exp15_Hybrid", os.path.join(base_exp15, "pred_dev_vllm_hybrid.jsonl"), "retrieved_context_raw"),
        ("Baseline_Dense", os.path.join(base_baseline, "pred_ie/dense/musique/qwen/pred_retrieval.jsonl"), "ctxs"),
        ("Baseline_BM25", os.path.join(base_baseline, "pred_ie/bm25/musique/qwen/pred_retrieval.jsonl"), "ctxs")
    ]
    
    results = {}
    
    for label, path, field in configs:
        if os.path.exists(path):
            results[label] = calculate_ie_series_generic(path, gold_map, field, max_k=50)
        else:
            print(f"Error: File not found: {path}")
            results[label] = [0.0] * 50
            
    # 3. Write Report
    output_file = "/home/wjk/workplace/nq/ano-rag/IE_Comparison_Report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# MuSiQue Retrieval IE@K Comparison (Exp 15 vs Baseline)\n\n")
        f.write("| K | Exp15 Dense | Exp15 BM25 | Exp15 Hybrid | Baseline Dense | Baseline BM25 |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for k in range(1, 51):
            idx = k - 1
            row = f"| {k} |"
            for label, _, _ in configs:
                val = results[label][idx]
                row += f" {val:.4f} |"
            f.write(row + "\n")
            
    print(f"\nReport written to {output_file}")
    
    # Also print to stdout for immediate checking
    print("\nPreview (K=1, 5, 10, 20, 50):")
    header = "K\t" + "\t".join([c[0] for c in configs])
    print(header)
    for k in [1, 5, 10, 20, 50]:
        idx = k - 1
        vals = [f"{results[c[0]][idx]:.4f}" for c in configs]
        print(f"{k}\t" + "\t".join(vals))

if __name__ == "__main__":
    main()
