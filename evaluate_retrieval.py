import json
import os
import re
from pathlib import Path
from typing import Dict, Set, List, Tuple

def load_ground_truth(doc_pool_path: str) -> Tuple[Set[Tuple[str, int]], Set[str]]:
    """
    Returns:
    1. relevant_chunks: Set of (mapped_id, chunk_index) where support=1
    2. relevant_docs: Set of mapped_id where at least one chunk has support=1
    """
    relevant_chunks = set()
    relevant_docs = set()
    
    with open(doc_pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for i, item in enumerate(data):
        if item.get("support") == 1:
            mapped_id = item.get("mapped_id")
            if mapped_id:
                relevant_chunks.add((mapped_id, i))
                relevant_docs.add(mapped_id)
                
    return relevant_chunks, relevant_docs

def evaluate_run(run_path: str, relevant_chunks: Set[Tuple[str, int]], relevant_docs: Set[str], is_raptor: bool = False):
    retrieval_file = Path(run_path) / "retrieval.jsonl"
    if not retrieval_file.exists():
        print(f"Skipping {run_path}: retrieval.jsonl not found")
        return None

    # Load retrieval results
    q_retrieved = {}
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                qid = entry.get("id")
                retrieved = entry.get("retrieved", [])
                q_retrieved[qid] = retrieved
            except:
                continue
                
    # Calculate metrics
    k_list = [1, 3, 5, 10]
    hits = {k: 0 for k in k_list}
    total = 0
    
    for qid, retrieved_items in q_retrieved.items():
        if qid not in relevant_docs:
            pass
            
        total += 1
        
        # Check hits at each K
        for k in k_list:
            top_k = retrieved_items[:k]
            is_hit = False
            
            for item in top_k:
                passage_id = item.get("passage_id", "")
                doc_id = item.get("doc_id", "")
                
                # Check for FiD format (mirage/UUID::INDEX)
                if doc_id and str(doc_id).startswith("mirage/"):
                    clean_id = str(doc_id).replace("mirage/", "")
                    parts = clean_id.split("::")
                    if len(parts) >= 2:
                        retrieved_mapped_id = parts[0]
                        try:
                            chunk_idx = int(parts[1])
                            if (retrieved_mapped_id, chunk_idx) in relevant_chunks:
                                is_hit = True
                                break
                        except ValueError:
                            pass
                    # If FiD format detected, we processed it. If not hit, continue loop?
                    # Wait, if we matched format but not hit, we should continue to next item in top_k?
                    # Yes, the loop is over items. break breaks inner loop (item loop).
                    # We want to break item loop if hit found.
                    if is_hit: break
                    continue 

                if is_raptor:
                    # Raptor format: MAPPED_ID::chunk_0 (or similar)
                    if "::" in passage_id:
                        retrieved_mapped_id = passage_id.split("::")[0]
                    else:
                        retrieved_mapped_id = passage_id 
                        
                    if retrieved_mapped_id == qid and qid in relevant_docs:
                        is_hit = True
                        break
                else:
                    # Standard format: MAPPED_ID::INDEX::CHUNK
                    parts = passage_id.split("::")
                    if len(parts) >= 2:
                        retrieved_mapped_id = parts[0]
                        try:
                            chunk_idx = int(parts[1])
                            if (retrieved_mapped_id, chunk_idx) in relevant_chunks:
                                is_hit = True
                                break
                        except ValueError:
                            pass
                            
            if is_hit:
                hits[k] += 1
                
    metrics = {f"R@{k}": (hits[k] / total) if total > 0 else 0.0 for k in k_list}
    metrics["count"] = total
    return metrics

def main():
    doc_pool_path = "data/mirage_sample_200/doc_pool.json"
    relevant_chunks, relevant_docs = load_ground_truth(doc_pool_path)
    
    runs = [
        ("Vanilla RAG", "result/mirage_vanilla_rag_011", False),
        ("FiD RAG", "result/mirage_fid_rag_200", False),
        ("Simple Self-RAG", "result/mirage_simple_selfrag_009", False),
        ("Simple Raptor", "result/mirage_raptor_run_200", True),
        ("Simple GraphRAG", "result/mirage_graphrag_run_200", False) 
    ]
    
    results = {}
    for name, path, is_raptor in runs:
        metrics = evaluate_run(path, relevant_chunks, relevant_docs, is_raptor)
        if metrics:
            results[name] = metrics
            
    # Print Markdown Table
    print("| Baseline | R@1 | R@3 | R@5 | R@10 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['R@1']:.4f} | {metrics['R@3']:.4f} | {metrics['R@5']:.4f} | {metrics['R@10']:.4f} |")
        
    # Write to file
    with open("baseline_results_200.md", "w") as f:
        f.write("# Baseline Retrieval Performance (MIRAGE Sample 200)\n\n")
        f.write("| Baseline | R@1 | R@3 | R@5 | R@10 | Count |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['R@1']:.4f} | {metrics['R@3']:.4f} | {metrics['R@5']:.4f} | {metrics['R@10']:.4f} | {metrics['count']} |\n")

if __name__ == "__main__":
    main()
