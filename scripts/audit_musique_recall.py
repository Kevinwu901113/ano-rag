import json
import sys

def analyze_recall(file_path):
    print(f"Scanning {file_path} for perfect recall cases (R@10=1.0)...")
    gold_counts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        perfect_count = 0
        for line in f:
            try:
                row = json.loads(line)
            except:
                continue
            
            # Extract Gold Titles
            gold_sp = row.get("gold_sp") or []
            gold_titles = set()
            for item in gold_sp:
                if isinstance(item, list) and len(item) >= 1:
                    title = str(item[0]).strip()
                    if title:
                        gold_titles.add(title)
            
            if not gold_titles:
                continue

            # Extract Retrieved Titles (Top 10)
            context = row.get("retrieved_context_raw") or []
            retrieved_titles = []
            for item in context:
                title = item.get("title") or item.get("doc_title")
                if title:
                    retrieved_titles.append(str(title).strip())
            
            top_10 = retrieved_titles[:10]
            top_10_set = set(top_10)
            
            # Check Recall
            hits = [t for t in gold_titles if t in top_10_set]
            recall = len(hits) / len(gold_titles)
            
            num_gold = len(gold_titles)
            
            # Show distribution stats
            if num_gold not in gold_counts:
                gold_counts[num_gold] = {"total": 0, "perfect": 0}
            gold_counts[num_gold]["total"] += 1
            if recall == 1.0:
                gold_counts[num_gold]["perfect"] += 1

            if recall == 1.0:
                perfect_count += 1
                if perfect_count <= 5: # Show first 5 examples
                    print(f"\n[Case #{perfect_count}]")
                    print(f"ID: {row.get('id')}")
                    print(f"Question: {row.get('question')}")
                    print(f"Gold Titles ({len(gold_titles)}): {list(gold_titles)}")
                    
                    # Show ranks
                    print("Gold Title Ranks:")
                    for gt in gold_titles:
                        try:
                            rank = retrieved_titles.index(gt) + 1
                        except ValueError:
                            rank = "Not in raw list"
                        print(f"  - '{gt}': Rank {rank}")
                        
                    print(f"Top 10 Retrieved: {top_10}")

            count += 1
    
    print(f"\nTotal scanned: {count}")
    print(f"Perfect Recall Cases: {perfect_count} ({perfect_count/count:.2%})")
    
    print("\nGold SP Count Distribution:")
    print("| Num Gold | Total Samples | Perfect Recall Count | Perfect Recall Rate |")
    print("|---|---|---|---|")
    for k in sorted(gold_counts.keys()):
        stats = gold_counts[k]
        rate = stats["perfect"] / stats["total"]
        print(f"| {k} | {stats['total']} | {stats['perfect']} | {rate:.2%} |")

if __name__ == "__main__":
    gold_counts = {}

    if len(sys.argv) < 2:
        print("Usage: python3 scripts/audit_musique_recall.py <jsonl_file>")
        sys.exit(1)
    analyze_recall(sys.argv[1])
