import argparse
import json
import collections
import string
import re
import numpy as np
from typing import List, Dict, Any, Tuple

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

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--output_file", required=False)
    args = parser.parse_args()

    # Load Gold
    gold_data = {}
    print(f"Loading gold from {args.gold_file}...")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        # Check if array or jsonl
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            gold_list = json.load(f)
            gold_data = {str(item.get('_id') or item.get('id')): item for item in gold_list}
        else:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    gold_data[str(item.get('_id') or item.get('id'))] = item
                except:
                    pass

    # Load Pred
    print(f"Loading preds from {args.pred_file}...")
    preds = load_jsonl(args.pred_file)
    
    metrics = {'em': [], 'f1': []}
    
    for pred in preds:
        qid = str(pred.get('_id') or pred.get('id'))
        if qid not in gold_data:
            continue
            
        gold_entry = gold_data[qid]
        
        # Handle multiple possible answers
        gold_answers = gold_entry.get('answer', [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        elif not isinstance(gold_answers, list):
             # Fallback for weird formats
             gold_answers = [str(gold_answers)]
        
        # Also check 'references' field (HotpotQA style)
        if 'references' in gold_entry:
             gold_answers = gold_entry['references']
             
        # Support 'pred' key (baseline format), 'short_answer', 'prediction', 'answer'
        pred_answer = pred.get('short_answer', pred.get('prediction', pred.get('pred', pred.get('answer', ''))))
        if isinstance(pred_answer, dict): # Sometimes it's a dict?
             pred_answer = pred_answer.get('text', '')
        
        # Calculate max over all valid gold answers
        best_em = 0
        best_f1 = 0
        
        for gold_ans in gold_answers:
            em = exact_match_score(pred_answer, gold_ans)
            f1, _, _ = f1_score(pred_answer, gold_ans)
            
            if em > best_em: best_em = em
            if f1 > best_f1: best_f1 = f1
            
        metrics['em'].append(float(best_em))
        metrics['f1'].append(best_f1)
        
    results = {
        'em': np.mean(metrics['em']) if metrics['em'] else 0.0,
        'f1': np.mean(metrics['f1']) if metrics['f1'] else 0.0,
        'count': len(metrics['em'])
    }
    
    print(json.dumps(results, indent=2))
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
