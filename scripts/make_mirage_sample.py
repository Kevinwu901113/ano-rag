import json
import os
import argparse
from pathlib import Path
import random

def main():
    parser = argparse.ArgumentParser(description="Create a sample from MIRAGE dataset")
    parser.add_argument("--src-dir", default="data/mirage", help="Source directory containing dataset.json and doc_pool.json")
    parser.add_argument("--out-dir", default="data/mirage_sample_200", help="Output directory")
    parser.add_argument("--size", type=int, default=200, help="Number of QA pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {src}/dataset.json...")
    with open(src / "dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if len(dataset) < args.size:
        print(f"Warning: Dataset size {len(dataset)} is smaller than requested sample size {args.size}. Using full dataset.")
        sample_dataset = dataset
    else:
        # random.seed(args.seed)
        # sample_dataset = random.sample(dataset, args.size)
        # Prefer taking the first N for deterministic behavior if order matters, 
        # or just simple slicing if we don't strictly need random.
        # Given "like the 20 sample", maybe first N is better if the original 20 were first 20.
        # But to be safe and get a good distribution, let's just take first N for reproducibility without random shuffle unless needed.
        # Actually, let's just take the first N to keep it simple and consistent.
        sample_dataset = dataset[:args.size]

    print(f"Selected {len(sample_dataset)} queries.")
    
    # Collect query IDs
    query_ids = set()
    for item in sample_dataset:
        qid = item.get("query_id")
        if qid:
            query_ids.add(qid)
            
    print(f"Collected {len(query_ids)} unique query IDs.")

    print(f"Loading doc pool from {src}/doc_pool.json...")
    with open(src / "doc_pool.json", "r", encoding="utf-8") as f:
        doc_pool = json.load(f)
        
    print(f"Filtering doc pool (total {len(doc_pool)} docs)...")
    sample_doc_pool = []
    
    # In MIRAGE, doc_pool items are linked to queries via mapped_id.
    # We include docs that belong to our selected queries.
    # Note: doc_pool might contain distractors (negative samples) that are NOT linked to any query?
    # Or maybe distractors are linked to specific queries too?
    # In the sample 20, all docs had a mapped_id.
    # Some had support=1, some support=0.
    
    related_docs_count = 0
    for doc in doc_pool:
        mapped_id = doc.get("mapped_id")
        if mapped_id in query_ids:
            sample_doc_pool.append(doc)
            related_docs_count += 1
            
    print(f"Selected {len(sample_doc_pool)} documents related to the sampled queries.")
    
    # Save results
    print(f"Saving to {out}...")
    with open(out / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
        
    with open(out / "doc_pool.json", "w", encoding="utf-8") as f:
        json.dump(sample_doc_pool, f, indent=2, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    main()
