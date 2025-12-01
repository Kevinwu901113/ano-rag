import json
import csv
import sys
import re
import string
import glob
import os
from collections import Counter
from pathlib import Path

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

def evaluate_file(dataset_map, results_path):
    print(f"Evaluating {results_path}...")
    results = []
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    results.append((row[0], row[1]))
    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None

    total_count = 0
    exact_match_total = 0
    f1_total = 0
    
    for query_raw, model_output in results:
        query_norm = normalize_answer(query_raw)
        
        if query_norm not in dataset_map:
            continue
            
        valid_answers = dataset_map[query_norm]
        
        em = metric_max_over_ground_truths(exact_match_score, model_output, valid_answers)
        f1 = metric_max_over_ground_truths(f1_score, model_output, valid_answers)
        
        exact_match_total += em
        f1_total += f1
        total_count += 1

    if total_count == 0:
        return {
            "file": os.path.basename(results_path),
            "total": 0,
            "em": 0.0,
            "f1": 0.0
        }

    em_score = 100.0 * exact_match_total / total_count
    f1_score_avg = 100.0 * f1_total / total_count
    
    return {
        "file": os.path.basename(results_path),
        "total": total_count,
        "em": em_score,
        "f1": f1_score_avg
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_evaluate.py <dataset_path> <ans_dir> [output_md_path]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    ans_dir = sys.argv[2]
    output_md_path = sys.argv[3] if len(sys.argv) > 3 else "evaluation_report.md"

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    ground_truth_map = {}
    for item in dataset:
        query_norm = normalize_answer(item['query'])
        ground_truth_map[query_norm] = item['answer']
        
    print(f"Loaded {len(ground_truth_map)} queries from dataset.")

    # Find all tsv files in ans_dir
    result_files = glob.glob(os.path.join(ans_dir, "*.tsv"))
    result_files.sort()
    
    eval_results = []
    for res_file in result_files:
        res = evaluate_file(ground_truth_map, res_file)
        if res:
            eval_results.append(res)

    # Generate Markdown report
    md_content = "# Evaluation Report\n\n"
    md_content += f"**Dataset:** `{dataset_path}`\n"
    md_content += f"**Date:** {os.popen('date').read().strip()}\n\n"
    
    md_content += "| Method (File) | Total Queries | Exact Match (EM) | F1 Score |\n"
    md_content += "| :--- | :---: | :---: | :---: |\n"
    
    for res in eval_results:
        md_content += f"| {res['file']} | {res['total']} | {res['em']:.2f} | {res['f1']:.2f} |\n"
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"\nReport generated at {output_md_path}")
    print(md_content)

if __name__ == "__main__":
    main()
