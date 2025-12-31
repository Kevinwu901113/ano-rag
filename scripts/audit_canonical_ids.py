import json
import sys
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_standard import load_gold_dataset, load_jsonl

def check_ids(dataset_name, gold_path, retrieval_path, dataset_type):
    print(f"\n=== {dataset_name} ID Audit ===")
    qa_gold, ret_gold, canonical_type = load_gold_dataset(Path(gold_path), dataset_type)
    
    ret_data = load_jsonl(Path(retrieval_path))
    
    for i, item in enumerate(ret_data[:5]): # Check first 5 (we have 3)
        qid = str(item["id"])
        print(f"\nQID: {qid}")
        
        gold_ids = ret_gold.get(qid, set())
        print(f"Gold IDs ({len(gold_ids)}): {list(gold_ids)[:5]}...")
        
        retrieved = item.get("retrieved", [])
        ret_ids = []
        for r in retrieved[:5]:
            # Use canonical type key
            if canonical_type == "passage_id":
                did = str(r.get("passage_id"))
            else:
                did = str(r.get("doc_id"))
                
            if dataset_type == "mirage" and did.startswith("mirage/"):
                did = did.replace("mirage/", "").split("::")[0]
            ret_ids.append(did)
            
        print(f"Top-5 Retrieved IDs: {ret_ids}")
        
        # Match check
        matches = [rid for rid in ret_ids if rid in gold_ids]
        print(f"Matches: {matches}")
        print(f"Hit@5: {len(matches) > 0}")

def main():
    # HotpotQA
    check_ids("HotpotQA", 
              "data/hotpotqa/dataset_distractor_200.json",
              "result_relrag/20251231/hotpotqa_sample_200/bm25/budget_4096/artifacts/retrieval.jsonl",
              "hotpotqa")
              
    # MuSiQue
    check_ids("MuSiQue", 
              "data/musique_sample/musique.jsonl",
              "result_relrag/20251231/musique_sample_200/bm25/budget_4096/artifacts/retrieval.jsonl",
              "musique")
              
    # MIRAGE
    check_ids("MIRAGE", 
              "data/mirage_sample_200/dataset.json",
              "result_relrag/20251231/mirage_sample_200/bm25/budget_4096/artifacts/retrieval.jsonl",
              "mirage")

if __name__ == "__main__":
    main()
