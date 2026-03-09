import json
import re
import os
import string
from collections import Counter, defaultdict
import numpy as np

# --- 1. Metrics Implementation (Standard SQuAD) ---
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_max_f1(prediction, gold_answers):
    max_f1 = 0.0
    for gold in gold_answers:
        max_f1 = max(max_f1, f1_score(prediction, gold))
    return max_f1

# --- 2. Load Gold Data ---
gold_path = "/home/wjk/workplace/nq/ano-rag/musique/data/musique_ans_v1.0_dev.jsonl"
gold_map = {} # id -> list of answers

print(f"Loading gold data from {gold_path}...")
with open(gold_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        qid = data['id']
        answers = [data['answer']] + data.get('aliases', [])
        # Ensure list of strings
        answers = [str(a) for a in answers if a]
        gold_map[qid] = answers
print(f"Loaded {len(gold_map)} gold samples.")

# --- 3. Evaluation Function ---
def evaluate_file(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return {}
    
    hop_scores = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data.get('id', '')
                pred = data.get('pred', '')
                
                if qid not in gold_map:
                    continue
                
                # Calculate F1
                score = get_max_f1(pred, gold_map[qid])
                
                # Get Hop
                hop_match = re.match(r'^(\d+)hop', qid)
                if hop_match:
                    hop = int(hop_match.group(1))
                    hop_scores[hop].append(score)
            except json.JSONDecodeError:
                continue
    
    # Average per hop
    avg_scores = {}
    for hop, scores in hop_scores.items():
        if scores:
            avg_scores[hop] = sum(scores) / len(scores)
    return avg_scores

# --- 4. Process Files ---
base_path = "/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154/pred"
files = {
    "bm25_qwen": os.path.join(base_path, "bm25/musique/qwen/pred.jsonl"),
    "bm25_deepseek": os.path.join(base_path, "bm25/musique/deepseek/pred.jsonl"),
    "dense_qwen": os.path.join(base_path, "dense/musique/qwen/pred.jsonl"),
    "dense_deepseek": os.path.join(base_path, "dense/musique/deepseek/pred.jsonl"),
}

results = {}
for name, path in files.items():
    print(f"Evaluating {name}...")
    results[name] = evaluate_file(path)
    print(f"  -> {results[name]}")

# --- 5. Aggregate ---
hops = [2, 3, 4]
best_bm25 = {}
best_dense = {}

print("\n--- Aggregation ---")
for hop in hops:
    # BM25 Best
    s1 = results["bm25_qwen"].get(hop, 0.0)
    s2 = results["bm25_deepseek"].get(hop, 0.0)
    best_bm25[hop] = max(s1, s2)
    
    # Dense Best
    d1 = results["dense_qwen"].get(hop, 0.0)
    d2 = results["dense_deepseek"].get(hop, 0.0)
    best_dense[hop] = max(d1, d2)
    
    print(f"Hop {hop}: Best BM25 = {best_bm25[hop]:.4f}, Best Dense = {best_dense[hop]:.4f}")

final_scores = {}
for hop in hops:
    final_scores[hop] = (best_bm25[hop] + best_dense[hop]) / 2

print("\n--- Final Results (Standard RAG) ---")
for hop in hops:
    print(f"Hop {hop}: {final_scores[hop]:.4f}")

# Generate LaTeX coordinates
coords = "".join([f"({h},{final_scores[h]:.4f})" for h in hops])
print(f"\nLaTeX Coordinates: {coords}")
