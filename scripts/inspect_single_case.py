import json
import sys

def main():
    target_id = "4hop2__103790_14670_8987_8529"
    file_path = "debug/musique_failures_detailed.jsonl"
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] == target_id:
                print(f"=== Question: {data['question']} ===")
                print(f"=== Gold Answer: {data['gold_answer']} ===")
                print(f"=== Prediction: {data['prediction']} ===")
                print("\n=== Gold Supporting Facts ===")
                gold_titles = {sp[0] for sp in data['gold_sp']}
                print(gold_titles)
                
                print("\n=== Retrieved Context (Top 10) ===")
                # Sort by score or use order if score is None/same? 
                # Usually list is already sorted by retriever.
                # But wait, musique_entry.py filters top_k from retrieved_context_topk.
                # Let's look at 'retrieved_context_raw' but limit to first 10
                
                context = data.get('retrieved_context_topk', [])
                if not context:
                    context = data.get('retrieved_context_raw', [])
                
                # Deduplicate by title for display clarity if needed, but let's just show top 10 items
                for i, ctx in enumerate(context[:10]):
                    is_gold = ctx['title'] in gold_titles
                    marker = "★" if is_gold else " "
                    print(f"[{i+1}] {marker} Title: {ctx['title']}")
                    print(f"    Text: {ctx['text'][:300]}...") # Truncate text
                    print("-" * 40)
                return

if __name__ == "__main__":
    main()
