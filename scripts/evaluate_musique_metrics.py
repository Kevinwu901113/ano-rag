
import argparse
import json
import csv
import sys
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

# Re-use the cleaning and normalization logic
def clean_prediction(text: str) -> str:
    """
    Clean the prediction text by removing common prefixes/suffixes 
    and extracting the actual answer from verbose outputs.
    """
    if not text:
        return ""
    text = str(text)
    
    # Handle "Answer: ..." (case insensitive)
    if "answer:" in text.lower():
        parts = re.split(r"answer\s*:", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate:
                text = candidate
    
    # Handle "The answer is ..."
    if "the answer is" in text.lower():
        parts = re.split(r"the answer is\s*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate:
                text = candidate

    return text.strip()

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

def load_musique_dataset(path: str) -> Dict[str, List[str]]:
    """Load MuSiQue dataset and return a map of id -> list of acceptable answers."""
    gt_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item['id'])
            
            # Extract valid answers
            # MuSiQue usually has "answer" (str) and "answer_aliases" (list of str)
            answers = []
            if "answer" in item and item["answer"]:
                answers.append(item["answer"])
            if "answer_aliases" in item and item["answer_aliases"]:
                answers.extend(item["answer_aliases"])
            
            # Deduplicate
            gt_map[qid] = list(set(answers))
    return gt_map

def evaluate_run(run_dir: Path, gt_map: Dict[str, List[str]]):
    """Evaluate a single run directory."""
    # Look for results file: pred.jsonl or musique_results.jsonl or qa.tsv
    # Priority: musique_results.jsonl (official) > pred.json (hotpot style) > qa.tsv
    
    preds = {}
    
    # 1. Try musique_results.jsonl (NDJSON with id, predicted_answer)
    preds_dir = run_dir / "preds"
    res_path = preds_dir / "musique_results.jsonl"
    if res_path.exists():
        with open(res_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    pid = str(obj.get("id"))
                    ans = obj.get("predicted_answer") or ""
                    preds[pid] = ans
                except:
                    pass
    
    # 2. Try pred.json (HotpotQA style: {"answer": {id: ans}, ...})
    if not preds:
        pred_path = preds_dir / "pred.json"
        if pred_path.exists():
            with open(pred_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "answer" in data:
                    preds = {str(k): str(v) for k, v in data["answer"].items()}

    # 3. Try qa.tsv (question \t answer) - Hard to map back to ID without order, 
    # but maybe we can assume order if IDs match dataset order? 
    # For safety, let's skip QA TSV if we want ID matching.
    
    if not preds:
        return None

    # Calculate metrics
    exact_match_total = 0
    f1_total = 0
    total_count = 0
    
    for pid, pred_text in preds.items():
        if pid not in gt_map:
            continue
            
        clean_pred = clean_prediction(pred_text)
        valid_answers = gt_map[pid]
        
        em = metric_max_over_ground_truths(exact_match_score, clean_pred, valid_answers)
        f1 = metric_max_over_ground_truths(f1_score, clean_pred, valid_answers)
        
        exact_match_total += em
        f1_total += f1
        total_count += 1
        
    if total_count == 0:
        return 0.0, 0.0, 0
        
    return exact_match_total / total_count, f1_total / total_count, total_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/musique_sample/musique.jsonl")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--output", default=None, help="Path to write metrics JSON.")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading GT from {dataset_path}...")
    gt_map = load_musique_dataset(str(dataset_path))
    print(f"Loaded {len(gt_map)} questions.")
    
    runs = []
    if args.work_dir:
        run_dirs = [Path(args.work_dir)]
    else:
        root_dir = Path(args.result_root)
        if not root_dir.exists():
            print(f"Result root not found: {root_dir}")
            return
        run_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    for d in run_dirs:
        res = evaluate_run(d, gt_map)
        if res:
            runs.append((d.name, res[0], res[1], res[2]))
    
    # Sort by F1
    runs.sort(key=lambda x: x[2], reverse=True)
    
    metrics = {name: {"EM": em, "AnswerF1": f1, "count": count} for name, em, f1, count in runs}
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    print("\n| Run | EM | F1 | Count |")
    print("| --- | --- | --- | --- |")
    for name, em, f1, count in runs:
        print(f"| {name} | {em:.3f} | {f1:.3f} | {count} |")

if __name__ == "__main__":
    main()
