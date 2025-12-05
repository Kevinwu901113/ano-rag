#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Build doc_pool from HotpotQA dataset")
    parser.add_argument("--output", default="data/hotpotqa/doc_pool.json", help="Output path for doc_pool.json")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace datasets cache dir")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA dataset (fullwiki)...")
    # Load both train and validation to get a comprehensive set of documents referenced in the dataset
    # Note: This is NOT the full Wikipedia, but the subset provided in the HotpotQA dataset objects (gold + distractors)
    # If the user wanted the ACTUAL full wiki dump, they would need to provide a path to it. 
    # Based on instructions, we extract from the dataset.
    dataset = load_dataset("hotpot_qa", "fullwiki", cache_dir=args.cache_dir)
    
    doc_pool = {}

    for split in ["train", "validation"]:
        print(f"Processing {split} split...")
        for item in tqdm(dataset[split]):
            # item['context'] is composed of [title, sentences]
            titles = item['context']['title']
            sentences_list = item['context']['sentences']
            
            for title, sentences in zip(titles, sentences_list):
                if title not in doc_pool:
                    # Join sentences to form the document text
                    text = " ".join(sentences)
                    doc_pool[title] = text

    print(f"Total unique documents collected: {len(doc_pool)}")
    
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        # Write as JSONL as per instruction example: {"doc_id": "...", "text": "..."}
        # The user example showed multiple JSON objects, which implies JSONL.
        # But the filename is .json. 
        # "data/mirage_sample/doc_pool.json" in mirage seems to be a list or dict or jsonl?
        # Let's check the user input again:
        # {"doc_id": "...", "text": "..."}
        # {"doc_id": "...", "text": "..."}
        # This is JSONL.
        
        for title, text in doc_pool.items():
            record = {
                "doc_id": title,
                "text": text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
