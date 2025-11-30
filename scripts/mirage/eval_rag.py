#!/usr/bin/env python3
import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter

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
        if isinstance(text, list):
            return " ".join(str(t) for t in text).lower()
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    # Relaxed EM: if prediction is a substring of ground truth or vice versa, count as match?
    # Standard SQuAD EM is strict string equality after normalization.
    # But here we have generative model.
    # Let's keep standard EM but maybe add a "contains" metric if needed.
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def contains_score(prediction, ground_truth):
    pred = normalize_answer(prediction)
    gt = normalize_answer(ground_truth)
    if not pred or not gt:
        return False
    return (pred in gt) or (gt in pred)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return 0
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset_path, predictions_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Load predictions from TSV
    # Format: question \t answer
    predictions = {}
    with open(predictions_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                question = parts[0]
                answer = parts[1]
                predictions[question] = answer

    f1 = exact_match = contains = total = 0
    for item in dataset:
        question = item.get('query') or item.get('question')
        if question not in predictions:
            continue
            
        total += 1
        ground_truths = item.get('answers', [])
        if not ground_truths and 'answer' in item:
             ground_truths = [item['answer']]
             
        prediction = predictions[question]
        
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        contains += metric_max_over_ground_truths(
            contains_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total if total > 0 else 0
    contains = 100.0 * contains / total if total > 0 else 0
    f1 = 100.0 * f1 / total if total > 0 else 0

    return {'exact_match': exact_match, 'contains': contains, 'f1': f1, 'total': total}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/mirage_sample/dataset.json')
    parser.add_argument('--pred', required=True)
    args = parser.parse_args()
    
    results = evaluate(args.dataset, args.pred)
    print(json.dumps(results, indent=2))
