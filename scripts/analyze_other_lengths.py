import json
from pathlib import Path
import numpy as np

def analyze_lengths(path_str, dataset_type):
    path = Path(path_str)
    with path.open("r") as f:
        # HotpotQA / 2Wiki format: [{"context": [["Title", ["sent1", "sent2"]], ...]}, ...]
        data = [json.loads(line) for line in f if line.strip()]
        
    lengths = []
    
    for item in data:
        context = item.get("context", [])
        for doc in context:
            # doc is [Title, [sentences]]
            if len(doc) >= 2 and isinstance(doc[1], list):
                # Join sentences to form paragraph
                text = " ".join(doc[1])
                tokens = len(text.split())
                lengths.append(tokens)
            
    lengths = np.array(lengths)
    
    print(f"=== {dataset_type} ({path.name}) ===")
    print(f"Total paragraphs: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    
    under_256 = np.sum(lengths < 256)
    ratio_256 = under_256 / len(lengths)
    print(f"Paragraphs < 256 tokens: {under_256} ({ratio_256:.2%})")
    
    under_64 = np.sum(lengths < 64)
    ratio_64 = under_64 / len(lengths)
    print(f"Paragraphs < 64 tokens: {under_64} ({ratio_64:.2%})")
    print("-" * 40)

if __name__ == "__main__":
    analyze_lengths("data/hotpot_dev_distractor_500_jsonl.jsonl", "HotpotQA")
    analyze_lengths("data/2wiki_dev_sample_500.jsonl", "2WikiMultiHopQA")
