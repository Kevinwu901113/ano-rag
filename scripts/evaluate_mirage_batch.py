
import argparse
import json
import csv
import sys
import re
import string
from collections import Counter
from pathlib import Path

# Import functions from evaluate_mirage.py
sys.path.append(".")
from evaluate_mirage import clean_prediction, normalize_answer, exact_match_score, f1_score, metric_max_over_ground_truths

def evaluate_file(dataset_path, results_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    ground_truth_map = {}
    for item in dataset:
        query_norm = normalize_answer(item['query'])
        ground_truth_map[query_norm] = item['answer']
        
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                results.append((row[0], row[1]))
    
    total_count = 0
    exact_match_total = 0
    f1_total = 0
    
    for query_raw, model_output in results:
        model_output = clean_prediction(model_output)
        query_norm = normalize_answer(query_raw)
        
        if query_norm not in ground_truth_map:
            continue
            
        valid_answers = ground_truth_map[query_norm]
        
        em = metric_max_over_ground_truths(exact_match_score, model_output, valid_answers)
        f1 = metric_max_over_ground_truths(f1_score, model_output, valid_answers)
        
        exact_match_total += em
        f1_total += f1
        total_count += 1

    if total_count == 0:
        return 0.0, 0.0, 0
        
    em_score = exact_match_total / total_count
    f1_score_avg = f1_total / total_count
    return em_score, f1_score_avg, total_count

def main():
    dataset_path = "data/mirage_sample/dataset.json"
    ans_dir = Path("ans")
    
    print("| Run | EM | F1 | Count |")
    print("| --- | --- | --- | --- |")
    
    results = []
    
    for results_file in sorted(ans_dir.glob("*20qa.tsv")):
        em, f1, count = evaluate_file(dataset_path, results_file)
        results.append((results_file.name, em, f1, count))
        
    # Sort by F1 desc
    results.sort(key=lambda x: x[2], reverse=True)
    
    for name, em, f1, count in results:
        print(f"| {name} | {em:.3f} | {f1:.3f} | {count} |")

if __name__ == "__main__":
    main()
