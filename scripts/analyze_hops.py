import json
import re
from collections import defaultdict

file_path = "/home/wjk/workplace/nq/ano-rag/result/deepseek_smoke10/relrag/hybrid_qa/hotpotqa/deepseek/pred_dev_openai_hybrid.jsonl"

hop_stats = defaultdict(list)

with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        sample_id = data.get('id', '')
        f1 = data.get('metrics', {}).get('f1', 0.0)
        
        hop_match = re.match(r'^(\d+)hop', sample_id)
        if hop_match:
            hop = int(hop_match.group(1))
            hop_stats[hop].append(f1)

print("Hop Performance (F1):")
for hop in sorted(hop_stats.keys()):
    scores = hop_stats[hop]
    avg_f1 = sum(scores) / len(scores) if scores else 0
    print(f"Hop {hop}: {avg_f1:.4f} (count: {len(scores)})")
