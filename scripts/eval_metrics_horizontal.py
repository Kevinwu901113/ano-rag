import argparse
import json
import numpy as np
import os
import sys
import math
from typing import List, Dict, Set, Tuple, Any
import string
import collections

def normalize_answer(s):
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
        return 0, 0, 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0, 0, 0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def get_gold_titles(gold_sp):
    # gold_sp structure: [[title, sent_id], ...]
    return set(item[0] for item in gold_sp)

def get_pred_titles_topk(pred_sp_topk, k):
    # pred_sp_topk structure: [[title, sent_id], ...]
    # We take the top k items, then extract titles.
    # Note: If multiple sentences from same doc are retrieved, they count as the same doc.
    # But for "Top K retrieval", we usually look at the top K *items* returned by the retriever.
    # If the retriever returns sentences, K=5 means top 5 sentences.
    # We check if the gold paragraphs are covered by these top 5 sentences.
    
    top_items = pred_sp_topk[:k]
    return [item[0] for item in top_items]

def calculate_paragraph_recall(retrieved_titles_list: List[str], gold_titles: Set[str]) -> float:
    if not gold_titles:
        return 0.0
    # retrieved_titles_list may contain duplicates if multiple sentences from same doc are retrieved
    # Standard Recall@K: Does the set of retrieved documents contain the gold documents?
    # Actually, usually Recall = (Relevant Retrieved) / (Total Relevant)
    
    unique_retrieved = set(retrieved_titles_list)
    hits = len(unique_retrieved & gold_titles)
    return hits / len(gold_titles)

def calculate_ie(retrieved_titles_list: List[str], gold_titles: Set[str], k: int) -> float:
    # IE@K = (Number of relevant items in top K) / K
    # Note: If multiple sentences from same relevant doc are retrieved, do they count multiple times?
    # Usually IE measures "precision" or "useful information density".
    # If I retrieve 5 sentences from the same relevant doc, is that 100% efficient?
    # Standard definition: IE@K = sum(is_relevant(item_i)) / K
    # Yes, if the unit of retrieval is sentence, and the sentence is relevant (belongs to relevant doc), it counts.
    
    relevant_count = 0
    for title in retrieved_titles_list:
        if title in gold_titles:
            relevant_count += 1
    return relevant_count / k

def calculate_ndcg(retrieved_titles_list: List[str], gold_titles: Set[str], k: int) -> float:
    # Standard NDCG.
    # Relevance = 1 if title in gold_titles, else 0.
    relevance = []
    seen_titles = set() # To handle deduplication if we want Doc-Level NDCG on Sentence Retrieval?
    # User memory: "Prevent double-counting 'same sentence/paragraph' even if retrieved multiple times."
    # Memory 03fk5nh3e4u87s728a5jwvfv2: "NDCG... deduplicate by (title, sent_id)... Prevent double-counting..."
    # Wait, if we are doing Paragraph Level metrics (Gold Title matching), should we deduplicate by Title?
    # "Retrieval metrics (Recall/IE/NDCG) must use Paragraph-Level logic (Gold Title matching)"
    # If I retrieve [DocA-Sent1, DocA-Sent2], and DocA is relevant.
    # Rank 1: DocA-Sent1 (Rel=1)
    # Rank 2: DocA-Sent2 (Rel=0? because we already saw DocA?)
    # Memory says: "For NDCG... use seen tracking... Prevent double-counting same sentence/paragraph".
    # But here we are matching Titles.
    # If the user wants Paragraph Level evaluation on Sentence Retrieval results:
    # Typically, the first time we see a relevant Doc, it's a hit. Subsequent sentences from same Doc are redundant for "Doc Retrieval", but maybe not for "Evidence Retrieval".
    # However, for "Paragraph Recall", we usually care about finding the Paragraph.
    # Let's assume strict deduplication for NDCG as per memory "seen tracking".
    
    for title in retrieved_titles_list:
        if title in gold_titles and title not in seen_titles:
            relevance.append(1)
            seen_titles.add(title)
        else:
            relevance.append(0)
            
    # Pad with 0 if fewer than k items
    relevance = relevance + [0] * (k - len(relevance))
    relevance = relevance[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / math.log2(i + 2)
        
    # IDCG
    num_gold = len(gold_titles)
    ideal_k = min(num_gold, k)
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def evaluate_file(filepath):
    print(f"Processing {filepath}...", file=sys.stderr)
    metrics = {
        'em': [], 'f1': [],
        'recall@2': [], 'recall@5': [],
        'ie@2': [], 'ie@5': [],
        'ndcg@2': [], 'ndcg@5': []
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                pred = json.loads(line)
            except:
                continue
                
            # QA
            references = pred.get('references', [])
            if not references:
                gold_answer = pred.get('gold_answer', pred.get('answer', ''))
                if isinstance(gold_answer, list):
                    references = gold_answer
                else:
                    references = [gold_answer]
            
            # Prediction
            pred_answer = pred.get('short_answer', pred.get('prediction', ''))
            if pred_answer is None: pred_answer = ""
            
            # Calculate Max over references
            metrics['em'].append(max([exact_match_score(pred_answer, ref) for ref in references]) if references else 0.0)
            metrics['f1'].append(max([f1_score(pred_answer, ref)[0] for ref in references]) if references else 0.0)
            
            # Retrieval
            gold_sp = pred.get('gold_sp', [])
            gold_titles = get_gold_titles(gold_sp)
            
            pred_sp_topk = pred.get('pred_sp_topk', [])
            if not pred_sp_topk:
                # Fallback to retrieved_context_raw if topk not found
                pred_sp_topk = pred.get('retrieved_context_topk', pred.get('retrieved_context_raw', []))
                
                # Convert dict to list if needed
                if pred_sp_topk and isinstance(pred_sp_topk[0], dict):
                    # Convert dict to [title, sent_id]
                    new_topk = []
                    for item in pred_sp_topk:
                        t = item.get('title', '')
                        s = item.get('sentence_idx', 0)
                        new_topk.append([t, s])
                    pred_sp_topk = new_topk
                elif not pred_sp_topk:
                     # Last fallback to pred_sp (which might be just titles or [title, id])
                     pred_sp_topk = pred.get('pred_sp', [])
            
            # Ensure items are lists/tuples
            # pred_sp_topk is [[title, id], ...]
            
            for k in [2, 5]:
                retrieved_titles = get_pred_titles_topk(pred_sp_topk, k)
                metrics[f'recall@{k}'].append(calculate_paragraph_recall(retrieved_titles, gold_titles))
                metrics[f'ie@{k}'].append(calculate_ie(retrieved_titles, gold_titles, k))
                metrics[f'ndcg@{k}'].append(calculate_ndcg(retrieved_titles, gold_titles, k))

    # Aggregate
    results = {}
    for k, v in metrics.items():
        results[k] = np.mean(v) if v else 0.0
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    
    # Headers
    headers = ['File', 'EM', 'F1', 'R@2', 'R@5', 'IE@2', 'IE@5', 'NDCG@2', 'NDCG@5']
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    
    for filepath in args.files:
        res = evaluate_file(filepath)
        fname = os.path.basename(os.path.dirname(filepath)) + "/" + os.path.basename(filepath)
        # Shorten filename for display
        if "pred_dev_" in fname:
            fname = fname.replace("pred_dev_", "")
        if ".jsonl" in fname:
            fname = fname.replace(".jsonl", "")
            
        row = [
            fname,
            f"{res['em']:.4f}",
            f"{res['f1']:.4f}",
            f"{res['recall@2']:.4f}",
            f"{res['recall@5']:.4f}",
            f"{res['ie@2']:.4f}",
            f"{res['ie@5']:.4f}",
            f"{res['ndcg@2']:.4f}",
            f"{res['ndcg@5']:.4f}"
        ]
        print("| " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
