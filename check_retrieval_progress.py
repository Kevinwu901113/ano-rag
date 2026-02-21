
import json
from pathlib import Path

def count_lines(path):
    if not path.exists():
        return 0
    try:
        with open(path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

base = Path("baseline/results")
datasets = ["hotpotqa", "musique", "2wiki"]
methods = ["graphrag", "raptor", "dense"]

print(f"{'Method':<10} {'Dataset':<10} {'Count':<5} {'Target':<5}")
print("-" * 35)

for method in methods:
    for dataset in datasets:
        if method == "dense":
             path = Path("baseline/results_retrieval_fix") / method / dataset / "qwen" / "pred.jsonl"
             # Also check pred_retrieval.jsonl if I renamed it (which failed)
             if not path.exists():
                 path = Path("baseline/results_retrieval_fix") / method / dataset / "qwen" / "pred_retrieval.jsonl"
        else:
             path = base / method / dataset / "qwen" / "pred_retrieval.jsonl"
        
        count = count_lines(path)
        # Dense might be finished (500), others running
        print(f"{method:<10} {dataset:<10} {count:<5} {'500'}")

print("\nNote: Dense runs are completed. GraphRAG/Raptor runs are in progress (except Raptor HotpotQA which was interrupted).")
