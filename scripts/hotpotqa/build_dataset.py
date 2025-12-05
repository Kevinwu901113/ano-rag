#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Build dataset.json from HotpotQA")
    parser.add_argument("--output", default="data/hotpotqa/dataset.json", help="Output path for dataset.json")
    parser.add_argument("--split", default="validation", choices=["train", "validation"], help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading HotpotQA dataset (fullwiki) split {args.split}...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split=args.split)
    
    if args.limit:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    output_data = []
    
    print("Processing...")
    for item in dataset:
        # Format: {"id": "...", "question": "...", "answer": "..."}
        record = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"]
        }
        output_data.append(record)

    print(f"Writing {len(output_data)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
            
    print("Done.")

if __name__ == "__main__":
    main()
