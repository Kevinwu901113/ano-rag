import json
import re
import glob
from collections import defaultdict

file_pattern = "/home/wjk/workplace/nq/ano-rag/result/musique_ablation_64/fold_*/fixed_64/predictions.jsonl"
files = glob.glob(file_pattern)

hop_stats = defaultdict(list)

print(f"Found {len(files)} files.")

for file_path in files:
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('id', '')
            # Try to get F1 from metrics
            metrics = data.get('metrics', {})
            f1 = metrics.get('f1', 0.0)
            
            hop_match = re.match(r'^(\d+)hop', sample_id)
            if hop_match:
                hop = int(hop_match.group(1))
                hop_stats[hop].append(f1)

print("Hop Performance (F1) - Standard RAG (Fixed 64):")
for hop in sorted(hop_stats.keys()):
    scores = hop_stats[hop]
    avg_f1 = sum(scores) / len(scores) if scores else 0
    print(f"Hop {hop}: {avg_f1:.4f} (count: {len(scores)})")
