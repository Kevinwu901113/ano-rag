import json
import os
import collections
import numpy as np
import glob
import re
import string
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
    relevance = []
    seen = set()
    for i in range(min(k, len(retrieved_items))):
        item = tuple(retrieved_items[i])
        if item in gold_set and item not in seen:
            relevance.append(1)
            seen.add(item)
        else:
            relevance.append(0)
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        
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
    hits = len(set(retrieved_k) & gold_set)
    return hits / len(gold_set)

def calculate_paragraph_recall(retrieved_items: List[Tuple], gold_titles: Set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    
    retrieved_k = retrieved_items[:k]
    pred_titles = set(item[0] for item in retrieved_k)
    hits = len(pred_titles & gold_titles)
    return hits / len(gold_titles)

def calculate_paragraph_ndcg(retrieved_items: List[Tuple], gold_titles: Set[str], k: int) -> float:
    relevance = []
    seen_titles = set()
    
    # Filter retrieved items to just top k
    retrieved_k = retrieved_items[:k]
    
    for item in retrieved_k:
        title = item[0]
        if title in gold_titles and title not in seen_titles:
            relevance.append(1)
            seen_titles.add(title)
        else:
            relevance.append(0)
            
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        
    # IDCG
    num_gold = len(gold_titles)
    ideal_k = min(num_gold, k)
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_ie_at_k(retrieved_items: List[Tuple], gold_titles: Set[str], k: int) -> float:
    effective_count = 0
    for i in range(min(k, len(retrieved_items))):
        title = retrieved_items[i][0] # (title, sent_id)
        if title in gold_titles:
            effective_count += 1
    return effective_count / k

def evaluate_run(pred_file: str, gold_data: Dict[str, Any]) -> Dict[str, float]:
    print(f"Reading {pred_file}...")
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = []
        for line in f:
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
    metrics = {
        'em': [],
        'f1': [],
        'recall@2': [],
        'recall@5': [],
        'ie@2': [],
        'ie@5': [],
        'ndcg@2': [],
        'ndcg@5': [],
    }
    
    count = 0
    for pred in preds:
        qid = pred.get('_id')
        if not qid:
            # Try 'id' if '_id' missing
            qid = pred.get('id')
            
        if qid not in gold_data:
            continue
            
        count += 1
        gold_entry = gold_data[qid]
        gold_answer = gold_entry.get('answer', '')
        
        # QA Metrics
        pred_answer = pred.get('short_answer', pred.get('answer', ''))
        
        em = exact_match_score(pred_answer, gold_answer)
        f1, _, _ = f1_score(pred_answer, gold_answer)
        
        metrics['em'].append(float(em))
        metrics['f1'].append(f1)
        
        # Retrieval Metrics
        gold_sp = set(tuple(x) for x in gold_entry.get('supporting_facts', []))
        gold_titles = set(x[0] for x in gold_sp)
        
        pred_sp_topk = pred.get('pred_sp_topk', [])
        if not pred_sp_topk:
             pred_sp_topk = pred.get('pred_sp', [])
             
        pred_sp_tuples = [tuple(x) for x in pred_sp_topk]
        
        # Using Paragraph-level metrics for all standard reports as requested
        metrics['recall@2'].append(calculate_paragraph_recall(pred_sp_tuples, gold_titles, 2))
        metrics['recall@5'].append(calculate_paragraph_recall(pred_sp_tuples, gold_titles, 5))
        metrics['ie@2'].append(calculate_ie_at_k(pred_sp_tuples, gold_titles, 2))
        metrics['ie@5'].append(calculate_ie_at_k(pred_sp_tuples, gold_titles, 5))
        metrics['ndcg@2'].append(calculate_paragraph_ndcg(pred_sp_tuples, gold_titles, 2))
        metrics['ndcg@5'].append(calculate_paragraph_ndcg(pred_sp_tuples, gold_titles, 5))
        
    aggregated = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    aggregated['count'] = count
    return aggregated

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/home/wjk/workplace/nq/ano-rag/result/experiment_2wiki")
    parser.add_argument("--gold_file", default="/home/wjk/workplace/nq/ano-rag/data/2wiki_dev_sample_500.jsonl")
    args = parser.parse_args()

    base_dir = args.base_dir
    gold_file = args.gold_file
    
    print(f"Loading gold data from {gold_file}...")
    gold_data = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        # Check if it's a json array or jsonl
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            gold_list = json.load(f)
            gold_data = {item['_id']: item for item in gold_list}
        else:
            for line in f:
                try:
                    item = json.loads(line)
                    gold_data[item['_id']] = item
                except:
                    pass
    
    print(f"Loaded {len(gold_data)} gold entries.")
    
    # Identify all pred files dynamically
    pred_files = glob.glob(os.path.join(base_dir, "pred_dev_*.jsonl"))
    results = {}
    
    for pred_file in pred_files:
        basename = os.path.basename(pred_file)
        config_name = basename.replace("pred_dev_", "").replace(".jsonl", "")
        
        print(f"Evaluating {config_name}...")
        results[config_name] = evaluate_run(pred_file, gold_data)
    
    # Generate Report
    report_path = os.path.join(base_dir, "evaluation_report_zh.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Experiment Evaluation Report\n\n")
        f.write("## 1. 评估概述\n")
        f.write(f"- **评估目录**: `{base_dir}`\n")
        f.write(f"- **金标数据**: `{gold_file}` (共 {len(gold_data)} 条)\n")
        f.write(f"- **评估指标**: F1, EM, Recall@2/5, IE@2/5, NDCG@2/5\n\n")
        
        f.write("## 2. 详细结果\n\n")
        f.write("| Config | F1 | EM | Recall@2 | Recall@5 | IE@2 | IE@5 | NDCG@2 | NDCG@5 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for config, metrics in results.items():
            f.write(f"| {config} | {metrics['f1']:.4f} | {metrics['em']:.4f} | {metrics['recall@2']:.4f} | {metrics['recall@5']:.4f} | {metrics['ie@2']:.4f} | {metrics['ie@5']:.4f} | {metrics['ndcg@2']:.4f} | {metrics['ndcg@5']:.4f} |\n")

                
        f.write("\n## 3. 结果分析\n")
        
        # Filter valid results (e.g. count > 10)
        valid_results = {k: v for k, v in results.items() if v['count'] > 10}
        
        if valid_results:
            best_f1_config = max(valid_results.keys(), key=lambda x: valid_results[x]['f1'])
            best_recall_config = max(valid_results.keys(), key=lambda x: valid_results[x]['ndcg@5'])
            
            f.write(f"- **最佳 QA 配置 (N>10)**: `{best_f1_config}` (F1: {valid_results[best_f1_config]['f1']:.4f})\n")
            f.write(f"- **最佳 检索 配置 (N>10)**: `{best_recall_config}` (NDCG@5: {valid_results[best_recall_config]['ndcg@5']:.4f})\n")
            
            # Compare OpenAI vs VLLM if possible
            openai_f1 = [valid_results[k]['f1'] for k in valid_results if 'openai' in k]
            vllm_f1 = [valid_results[k]['f1'] for k in valid_results if 'vllm' in k]
            
            if openai_f1 and vllm_f1:
                avg_openai = np.mean(openai_f1)
                avg_vllm = np.mean(vllm_f1)
                f.write(f"- **模型对比**: OpenAI 平均 F1 ({avg_openai:.4f}) vs VLLM 平均 F1 ({avg_vllm:.4f})\n")
        else:
            f.write("- **警告**: 没有发现样本数 > 10 的有效实验结果。\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
