import sys
import os
import json
import csv
from pathlib import Path
from collections import Counter
import re
import string

# --- F1/EM Logic ---
def normalize_answer(s):
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
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_qa(qa_path, ground_truth_map):
    results = []
    if not os.path.exists(qa_path):
        return {"EM": 0.0, "F1": 0.0, "Count": 0}
        
    with open(qa_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                results.append((row[0], row[1]))
                
    total = 0
    em_total = 0
    f1_total = 0
    
    for query_raw, model_output in results:
        query_norm = normalize_answer(query_raw)
        if query_norm not in ground_truth_map:
            continue
        valid_answers = ground_truth_map[query_norm]
        em_total += metric_max_over_ground_truths(exact_match_score, model_output, valid_answers)
        f1_total += metric_max_over_ground_truths(f1_score, model_output, valid_answers)
        total += 1
        
    if total == 0:
        return {"EM": 0.0, "F1": 0.0, "Count": 0}
        
    return {
        "EM": 100.0 * em_total / total,
        "F1": 100.0 * f1_total / total,
        "Count": total
    }

# --- Recall Logic ---
def load_retrieval_ground_truth(doc_pool_path):
    relevant_chunks = set()
    relevant_docs = set()
    try:
        with open(doc_pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if item.get("support") == 1:
                mapped_id = item.get("mapped_id")
                if mapped_id:
                    relevant_chunks.add((mapped_id, i))
                    relevant_docs.add(mapped_id)
    except Exception as e:
        print(f"Error loading doc pool: {e}")
    return relevant_chunks, relevant_docs

def evaluate_retrieval(run_path, relevant_chunks, relevant_docs):
    retrieval_file = Path(run_path) / "retrieval.jsonl"
    if not retrieval_file.exists():
        return None
        
    q_retrieved = {}
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                qid = entry.get("id")
                retrieved = entry.get("retrieved", [])
                q_retrieved[qid] = retrieved
            except:
                continue
                
    k_list = [1, 3, 5, 10]
    hits = {k: 0 for k in k_list}
    total = 0
    
    for qid, retrieved_items in q_retrieved.items():
        # Using evaluate_retrieval.py logic loosely
        if qid not in relevant_docs:
            pass 

        total += 1
        for k in k_list:
            top_k = retrieved_items[:k]
            is_hit = False
            for item in top_k:
                passage_id = item.get("passage_id", "")
                doc_id = item.get("doc_id", "")
                
                # Check 1: Standard Mirage format (mapped_id::index)
                clean_id = str(doc_id).replace("mirage/", "")
                parts = clean_id.split("::")
                if len(parts) >= 2:
                    retrieved_mapped_id = parts[0]
                    try:
                        chunk_idx = int(parts[1])
                        if (retrieved_mapped_id, chunk_idx) in relevant_chunks:
                            is_hit = True
                    except: pass
                
                # Check 2: Raptor or simple format (mapped_id only, or mapped_id::...)
                if not is_hit:
                    # Try passage_id
                    p_str = str(passage_id)
                    if "::" in p_str:
                        retrieved_mapped_id = p_str.split("::")[0]
                    else:
                        retrieved_mapped_id = p_str
                    
                    if retrieved_mapped_id == qid and qid in relevant_docs:
                         is_hit = True
                    
                    # Also check if doc_id matches qid directly (some baselines might use doc_id as mapped_id)
                    if str(doc_id) == qid and qid in relevant_docs:
                        is_hit = True

                if is_hit: break
            if is_hit: hits[k] += 1
            
    metrics = {f"R@{k}": (hits[k] / total) if total > 0 else 0.0 for k in k_list}
    return metrics

def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate_all_results.py <dataset_path> <doc_pool_path> <result_root>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    doc_pool_path = sys.argv[2]
    result_root = Path(sys.argv[3])
    
    # Load QA Ground Truth
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    qa_ground_truth = {}
    for item in dataset:
        q_text = item.get("query") or item.get("question")
        if q_text:
            query_norm = normalize_answer(q_text)
            qa_ground_truth[query_norm] = item.get("answer")
        
    # Load Retrieval Ground Truth
    relevant_chunks, relevant_docs = load_retrieval_ground_truth(doc_pool_path)
    
    print(f"| Run | EM | F1 | R@1 | R@3 | R@5 | R@10 | Count |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for run_dir in sorted(result_root.iterdir()):
        if not run_dir.is_dir(): continue
        # Filter for relevant directories (starts with mirage_)
        if not run_dir.name.startswith("mirage_"): continue

        # Evaluate QA
        qa_metrics = evaluate_qa(run_dir / "qa.tsv", qa_ground_truth)
        
        # Evaluate Retrieval
        ret_metrics = evaluate_retrieval(run_dir, relevant_chunks, relevant_docs)
        if not ret_metrics:
            ret_metrics = {"R@1": 0, "R@3": 0, "R@5": 0, "R@10": 0}
            
        print(f"| {run_dir.name} | {qa_metrics['EM']:.2f} | {qa_metrics['F1']:.2f} | "
              f"{ret_metrics['R@1']:.4f} | {ret_metrics['R@3']:.4f} | {ret_metrics['R@5']:.4f} | {ret_metrics['R@10']:.4f} | {qa_metrics['Count']} |")

if __name__ == "__main__":
    main()
