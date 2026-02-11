import json
import numpy as np
import collections
import string
import os
import glob
from typing import List, Dict, Tuple, Set

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

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_paragraph_ndcg(retrieved_titles: List[str], gold_titles: Set[str], k: int) -> float:
    relevance = []
    seen = set()
    
    # Take top k
    current_k_titles = retrieved_titles[:k]
    
    for title in current_k_titles:
        # Deduplicate within the retrieved list for NDCG calculation?
        # Standard NDCG usually doesn't penalize duplicates in the ranking list itself unless we explicitly want to.
        # But if we retrieve the same doc twice, it shouldn't count as two relevant hits for "set" based relevance.
        # However, usually retrieved_context is unique by doc_id/chunk_id. 
        # If we have multiple chunks from same doc, does it count?
        # Memory says: "Evaluation Logic: STRICTLY deduplicate retrieved items before calculating Recall/NDCG... Prevent double-counting 'same sentence/paragraph'"
        # Since we compare TITLES, multiple chunks from same Doc Title = Duplicate.
        # So we should track 'seen' in the ranking.
        
        if title in gold_titles and title not in seen:
            relevance.append(1)
            seen.add(title)
        else:
            relevance.append(0)
            # If title is in gold_titles but ALREADY seen, it's a duplicate relevant retrieval.
            # Should we count it as 0? Yes, to avoid double counting.
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        
    num_gold = len(gold_titles)
    ideal_k = min(num_gold, k)
    
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_paragraph_recall(retrieved_titles: List[str], gold_titles: Set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    
    retrieved_k = retrieved_titles[:k]
    # Unique titles in top k
    pred_titles = set(retrieved_k)
    hits = len(pred_titles & gold_titles)
    return hits / len(gold_titles)

def calculate_paragraph_ie(retrieved_titles: List[str], gold_titles: Set[str], k: int) -> float:
    # IE@K = Precision@K (relevant items in top K / K)
    # But deduplicated? Usually Precision counts unique relevant items.
    retrieved_k = retrieved_titles[:k]
    
    # If we have duplicates in retrieved_k (e.g. 2 chunks from same doc), do they count as 1 hit or 2?
    # For "Paragraph Level", usually we care about "Did we retrieve the Paragraph?".
    # If we retrieve 2 chunks from same paragraph, it's 1 hit.
    # If we retrieve 2 chunks from DIFFERENT paragraphs but SAME Title (in Hotpot/Wiki, granularity is often Sentence or Paragraph).
    # In Hotpot, Title is the granularity for "Paragraph".
    # So 2 chunks from same Title = 1 hit.
    
    unique_hits = 0
    seen = set()
    for title in retrieved_k:
        if title in gold_titles and title not in seen:
            unique_hits += 1
            seen.add(title)
            
    return unique_hits / k

def evaluate_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    metrics_sum = {
        'f1': 0.0, 'em': 0.0,
        'recall@2': 0.0, 'recall@5': 0.0,
        'ie@2': 0.0, 'ie@5': 0.0,
        'ndcg@2': 0.0, 'ndcg@5': 0.0
    }
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # QA Metrics
            prediction = data.get('prediction') or data.get('generated_answer') or ""
            # Gold Answer: handle list or string
            # Prioritize references which is standard list of gold answers
            gold_answers = data.get('references') or data.get('gold_answer')
            
            # If not found, fallback to 'answer' only if it's likely Gold (risk of leakage if it's prediction)
            # In MuSiQue pred files, 'answer' is prediction. In Hotpot, it might be too.
            # Safe bet: if references/gold_answer missing, treat as missing gold.
            if gold_answers is None:
                 # Fallback: check if 'answer' exists and differs from 'prediction'? 
                 # Or just skip.
                 # Let's try to use 'answer' but only if 'references' is missing, 
                 # and hope it's not the prediction. 
                 # actually, in the observed Hotpot file, answer == prediction.
                 # So we MUST NOT use 'answer' key for gold if it matches prediction logic.
                 # However, if references is present, use it.
                 gold_answers = []
            
            if not isinstance(gold_answers, list):
                if gold_answers:
                    gold_answers = [gold_answers]
                else:
                    gold_answers = []
            
            # F1/EM
            best_f1 = 0.0
            best_em = 0.0
            for gold in gold_answers:
                if not gold: continue
                best_f1 = max(best_f1, f1_score(prediction, gold))
                if exact_match_score(prediction, gold):
                    best_em = 1.0
            
            metrics_sum['f1'] += best_f1
            metrics_sum['em'] += best_em
            
            # Retrieval Metrics
            # Gold Titles
            gold_sp = data.get('gold_sp', [])
            if not gold_sp:
                # Fallback for MuSiQue if gold_sp missing but gold_paragraphs exists?
                # Assume gold_sp is present as checked.
                gold_titles = set()
            else:
                gold_titles = set(item[0] for item in gold_sp)
            
            # Retrieved Titles
            retrieved_context = data.get('retrieved_context_raw', [])
            # Extract titles
            retrieved_titles = []
            for item in retrieved_context:
                t = item.get('title') or item.get('doc_title')
                if t:
                    retrieved_titles.append(t)
            
            # Calculate
            metrics_sum['recall@2'] += calculate_paragraph_recall(retrieved_titles, gold_titles, 2)
            metrics_sum['recall@5'] += calculate_paragraph_recall(retrieved_titles, gold_titles, 5)
            
            metrics_sum['ie@2'] += calculate_paragraph_ie(retrieved_titles, gold_titles, 2)
            metrics_sum['ie@5'] += calculate_paragraph_ie(retrieved_titles, gold_titles, 5)
            
            metrics_sum['ndcg@2'] += calculate_paragraph_ndcg(retrieved_titles, gold_titles, 2)
            metrics_sum['ndcg@5'] += calculate_paragraph_ndcg(retrieved_titles, gold_titles, 5)
            
            count += 1
            
    if count == 0:
        return None
        
    return {k: v / count for k, v in metrics_sum.items()}

def main():
    dirs = [
        ("/home/wjk/workplace/nq/ano-rag/result/musique_experiment_10half", "MuSiQue"),
        ("/home/wjk/workplace/nq/ano-rag/result/experiment_24half", "HotpotQA"),
        ("/home/wjk/workplace/nq/ano-rag/result/experiment_2wiki_3half", "2Wiki")
    ]
    
    modes = ["bm25", "dense", "hybrid"]
    
    report_lines = []
    report_lines.append("# 实验评估报告 (Evaluation Report)")
    report_lines.append("")
    
    for dir_path, dataset_name in dirs:
        report_lines.append(f"## {dataset_name} ({os.path.basename(dir_path)})")
        report_lines.append("| Metric | BM25 | Dense | Hybrid |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        
        results = {}
        for mode in modes:
            file_name = f"pred_dev_{mode}.jsonl"
            file_path = os.path.join(dir_path, file_name)
            metrics = evaluate_file(file_path)
            results[mode] = metrics
            
        # Define metrics order
        metric_keys = ['f1', 'em', 'recall@2', 'recall@5', 'ie@2', 'ie@5', 'ndcg@2', 'ndcg@5']
        
        for key in metric_keys:
            row = f"| {key} |"
            for mode in modes:
                val = results[mode]
                if val:
                    row += f" {val[key]:.4f} |"
                else:
                    row += " N/A |"
            report_lines.append(row)
        report_lines.append("")
        
    # Write to file
    output_file = "/home/wjk/workplace/nq/ano-rag/experiment_evaluation_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report written to {output_file}")
    print(open(output_file).read())

if __name__ == "__main__":
    main()
