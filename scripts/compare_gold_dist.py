import json
import sys
from collections import Counter

def get_gold_sp_counts(path):
    print(f"Analyzing {path}...")
    counts = Counter()
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                # Count supporting paragraphs
                sp_count = 0
                for p in row.get('paragraphs', []):
                    if p.get('is_supporting'):
                        sp_count += 1
                
                counts[sp_count] += 1
                total += 1
            except:
                continue
    return counts, total

def compare_distributions(subset_path, full_path):
    sub_counts, sub_total = get_gold_sp_counts(subset_path)
    full_counts, full_total = get_gold_sp_counts(full_path)
    
    print("\n[Distribution Comparison]")
    print(f"{'Gold SP Count':<15} | {'Subset (466)':<15} | {'Full Set (2417)':<15} | {'Diff':<10}")
    print("-" * 65)
    
    all_keys = sorted(set(sub_counts.keys()) | set(full_counts.keys()))
    
    for k in all_keys:
        sub_pct = sub_counts[k] / sub_total * 100 if sub_total else 0
        full_pct = full_counts[k] / full_total * 100 if full_total else 0
        diff = sub_pct - full_pct
        
        print(f"{k:<15} | {sub_pct:>6.2f}% ({sub_counts[k]}) | {full_pct:>6.2f}% ({full_counts[k]}) | {diff:>+6.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/compare_gold_dist.py <subset_file> <full_file>")
        sys.exit(1)
        
    compare_distributions(sys.argv[1], sys.argv[2])
