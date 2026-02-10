import json
import sys

def main():
    target_id = "4hop2__103790_14670_8987_8529"
    file_path = "debug/musique_failures_detailed.jsonl"
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] == target_id:
                context = data.get('retrieved_context_topk', []) or data.get('retrieved_context_raw', [])
                for ctx in context:
                    if ctx['title'] == "Josip Broz Tito":
                        print(f"=== Full Text of Chunk: {ctx['title']} ===")
                        print(ctx['text'])
                        print("===")
                        return

if __name__ == "__main__":
    main()
