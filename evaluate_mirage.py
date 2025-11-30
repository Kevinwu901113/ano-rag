import json
import csv
import sys
import re
import string
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
        return text.lower()

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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset_path, results_path):
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create a map for quick lookup: query -> valid answers
    ground_truth_map = {}
    for item in dataset:
        # We normalize the query key to ensure matching, but keep raw answers
        query_norm = normalize_answer(item['query'])
        ground_truth_map[query_norm] = item['answer']
        
    print(f"Loaded {len(ground_truth_map)} queries from dataset.")

    # Load results
    print(f"Loading results from {results_path}...")
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                results.append((row[0], row[1]))
    
    print(f"Loaded {len(results)} results.")

    total_count = 0
    exact_match_total = 0
    f1_total = 0
    missing_in_dataset = 0

    for query_raw, model_output in results:
        query_norm = normalize_answer(query_raw)
        
        if query_norm not in ground_truth_map:
            missing_in_dataset += 1
            continue
            
        valid_answers = ground_truth_map[query_norm]
        
        em = metric_max_over_ground_truths(exact_match_score, model_output, valid_answers)
        f1 = metric_max_over_ground_truths(f1_score, model_output, valid_answers)
        
        exact_match_total += em
        f1_total += f1
        total_count += 1

    if total_count == 0:
        print("No matching queries found between dataset and results.")
        return

    em_score = 100.0 * exact_match_total / total_count
    f1_score_avg = 100.0 * f1_total / total_count
    
    print(f"\nEvaluation Results:")
    print(f"Total matched queries: {total_count}")
    print(f"Exact Match (EM): {em_score:.2f}")
    print(f"F1 Score: {f1_score_avg:.2f}")
    
    if missing_in_dataset > 0:
        print(f"Warning: {missing_in_dataset} queries in results were not found in the dataset.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_mirage.py <dataset_path> <results_path>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    results_file = sys.argv[2]
    evaluate(dataset_file, results_file)
