import json
import sys

def analyze(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"File: {path}")
    print(f"Total items: {len(data)}")
    
    if "doc_pool" in path:
        support_1 = sum(1 for x in data if x.get('support') == 1)
        support_0 = sum(1 for x in data if x.get('support') == 0)
        print(f"Support=1: {support_1}")
        print(f"Support=0: {support_0}")
        
        # Check unique mapped_ids
        mapped_ids = set(x.get('mapped_id') for x in data if x.get('mapped_id'))
        print(f"Unique mapped_ids: {len(mapped_ids)}")

print("--- Original Sample (20) ---")
analyze("data/mirage_sample/doc_pool.json")

print("\n--- New Sample (200) ---")
analyze("data/mirage_sample_200/doc_pool.json")
