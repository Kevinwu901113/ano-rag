import json
import os
import collections
import numpy as np
from typing import List, Dict, Tuple, Set, Any

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return ' '.join([t for t in text.split() if t not in ['a', 'an', 'the']])

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

import string
import re

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_ndcg(retrieved_items: List[Tuple], gold_set: Set[Tuple], k: int) -> float:
    """
    Calculate NDCG@k.
    retrieved_items: list of (title, sent_id)
    gold_set: set of (title, sent_id)
    """
    # Relevance list for retrieved items
    relevance = []
    for i in range(min(k, len(retrieved_items))):
        item = tuple(retrieved_items[i])
        relevance.append(1 if item in gold_set else 0)
    
    # Pad with 0 if fewer than k items retrieved (though usually we care about rank positions)
    # Actually standard DCG summation goes up to min(k, len(retrieved))
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2) # i+2 because log2(1)=0, rank starts at 1
        
    # IDCG: Ideal ranking has all true positives at the top
    num_gold = len(gold_set)
    ideal_k = min(num_gold, k)
    
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_recall(retrieved_items: List[Tuple], gold_set: Set[Tuple], k: int) -> float:
    if not gold_set:
        return 0.0
    
    retrieved_k = [tuple(x) for x in retrieved_items[:k]]
    hits = sum(1 for item in retrieved_k if item in gold_set)
    return hits / len(gold_set)

def evaluate_run(pred_file: str, gold_data: Dict[str, Any]) -> Dict[str, float]:
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = [json.loads(line) for line in f]
        
    metrics = {
        'em': [],
        'f1': [],
        'recall@2': [],
        'recall@5': [],
        'ndcg@10': []
    }
    
    for pred in preds:
        qid = pred['_id']
        if qid not in gold_data:
            continue
            
        gold_entry = gold_data[qid]
        gold_answer = gold_entry['answer']
        
        # QA Metrics
        # Try 'short_answer' first, then 'answer' (which might be prediction)
        pred_answer = pred.get('short_answer', pred.get('answer', ''))
        
        em = exact_match_score(pred_answer, gold_answer)
        f1, _, _ = f1_score(pred_answer, gold_answer)
        
        metrics['em'].append(float(em))
        metrics['f1'].append(f1)
        
        # Retrieval Metrics
        # gold_sp format: list of [title, sent_id]
        gold_sp = set(tuple(x) for x in gold_entry['supporting_facts'])
        
        # pred_sp_topk format: list of [title, sent_id]
        # Ensure elements are hashable tuples
        pred_sp_topk = pred.get('pred_sp_topk', [])
        # Sometimes pred_sp_topk might be missing or empty
        if not pred_sp_topk:
             # Fallback to pred_sp if topk missing? Usually topk is what we evaluate for retrieval
             pred_sp_topk = pred.get('pred_sp', [])
             
        pred_sp_tuples = [tuple(x) for x in pred_sp_topk]
        
        metrics['recall@2'].append(calculate_recall(pred_sp_tuples, gold_sp, 2))
        metrics['recall@5'].append(calculate_recall(pred_sp_tuples, gold_sp, 5))
        metrics['ndcg@10'].append(calculate_ndcg(pred_sp_tuples, gold_sp, 10))
        
    # Average
    aggregated = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    return aggregated

def main():
    base_dir = "/home/wjk/workplace/nq/ano-rag/result/experiment_22"
    gold_file = "/home/wjk/workplace/nq/ano-rag/data/hotpot_dev_distractor_500.json"
    
    print(f"Loading gold data from {gold_file}...")
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_list = json.load(f)
        gold_data = {item['_id']: item for item in gold_list}
        
    modes = ['bm25', 'dense', 'hybrid']
    results = {}
    
    for mode in modes:
        pred_file = os.path.join(base_dir, f"pred_dev_{mode}.jsonl")
        if os.path.exists(pred_file):
            print(f"Evaluating {mode}...")
            results[mode] = evaluate_run(pred_file, gold_data)
        else:
            print(f"Skipping {mode} (file not found)")
            
    # Generate Report
    report_path = os.path.join(base_dir, "evaluation_report_zh.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 22 评估报告\n\n")
        f.write("## 1. 评估概述\n")
        f.write(f"- **评估目录**: `{base_dir}`\n")
        f.write(f"- **金标数据**: `{gold_file}` (共 {len(gold_data)} 条)\n")
        f.write("- **评估指标**: F1, EM, Recall@2/5, NDCG@10\n\n")
        
        f.write("## 2. 详细结果\n\n")
        f.write("| 模型配置 (Retriever) | EM | F1 | Recall@2 | Recall@5 | NDCG@10 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for mode in modes:
            if mode in results:
                m = results[mode]
                f.write(f"| **{mode.capitalize()}** | {m['em']:.4f} | {m['f1']:.4f} | {m['recall@2']:.4f} | {m['recall@5']:.4f} | {m['ndcg@10']:.4f} |\n")
            else:
                f.write(f"| {mode.capitalize()} | - | - | - | - | - |\n")
                
        f.write("\n## 3. 结果分析\n")
        # Simple analysis generation
        best_f1_mode = max(results.keys(), key=lambda x: results[x]['f1']) if results else "None"
        best_recall_mode = max(results.keys(), key=lambda x: results[x]['recall@5']) if results else "None"
        
        f.write(f"- **QA 性能**: `{best_f1_mode}` 在 F1/EM 上表现最佳。\n")
        f.write(f"- **检索性能**: `{best_recall_mode}` 在 Recall@5 上表现最佳。\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
