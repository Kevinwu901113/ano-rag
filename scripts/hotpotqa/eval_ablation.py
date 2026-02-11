import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import math

# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import re
import string

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

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def calculate_recall_at_k(retrieved: List[Dict], answers: List[str], k: int) -> float:
    # Retrieved items are usually dicts with 'text' or 'evidence'
    # Answers is a list of acceptable answers
    # "Support partial match": check if ANY answer is contained in the text
    
    hits = 0
    top_k = retrieved[:k]
    
    # Check if we retrieved the answer in the top k chunks
    # This is "Hit@K" or "Recall@K" in the sense of "did we get it?"
    # If there are multiple answers, do we need to retrieve ALL of them?
    # Usually HotpotQA has one answer string (with aliases).
    # The requirement says "support multi-answer scenarios".
    # HotpotQA usually requires finding supporting facts.
    # But here we are evaluating "Sentence Splitting vs Chunking".
    # So we care if the *Answer String* is present in the retrieved chunks.
    
    # Let's assume we want to know if *at least one* valid answer is present in the union of top K chunks.
    # OR if we are measuring coverage of supporting facts?
    # The user said "Recall@2... support multi-answer scenarios under partial match".
    # I'll implement: For each answer in ground truth, is it present in the combined text of top K?
    # Then average over answers? Or just binary "is answer retrievable"?
    
    # Let's stick to standard "Answer Recall": Is the answer in the top K docs?
    
    combined_text = " ".join([(r.get("text") or r.get("evidence") or "") for r in top_k]).lower()
    
    for ans in answers:
        norm_ans = normalize_answer(ans)
        if norm_ans in combined_text:
            return 1.0
            
    return 0.0

def calculate_ndcg_at_k(retrieved: List[Dict], answers: List[str], k: int) -> float:
    # IDCG is 1.0 (assuming we want to find the answer at rank 1)
    # DCG = sum(rel_i / log2(i+1))
    # rel_i = 1 if chunk i contains answer, 0 otherwise.
    
    dcg = 0.0
    idcg = 0.0
    
    # We assume ideal case is finding answer at rank 1, 2, ... up to num valid chunks?
    # Or just rank 1? Usually for QA retrieval, we just want it to appear as early as possible.
    # Let's assume binary relevance.
    
    for i in range(min(len(retrieved), k)):
        chunk_text = (retrieved[i].get("text") or retrieved[i].get("evidence") or "").lower()
        rel = 0
        for ans in answers:
            if normalize_answer(ans) in chunk_text:
                rel = 1
                break
        
        dcg += rel / math.log2(i + 2)
        
    # Calculate IDCG
    # Ideal: All top ranks have relevance 1 until we run out of relevant chunks?
    # Or just 1 relevant chunk is enough?
    # If we treat this as "Answer Retrieval", usually there is 1 answer.
    # So IDCG for 1 relevant item at rank 1 is 1.0.
    # If we have multiple *supporting facts*, that's different.
    # User asked for "Recall... support multi-answer".
    # I'll assume we treat all chunks containing the answer as relevant.
    # But usually we only need to find it once.
    # Let's standard IDCG for "at least one relevant item exists": 1.0
    
    # Actually, if multiple chunks contain the answer, they are all relevant.
    # But having it twice doesn't help much.
    # Standard NDCG for QA usually assumes we want the answer at rank 1.
    
    # I will use a simple binary relevance IDCG=1.0.
    idcg = 1.0 
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_ie_at_k(retrieved: List[Dict], gold_titles: Set[str], k: int) -> float:
    # IE@K definition from HOTPOT_IE_EXPERIMENTS.md:
    # (1/K) * sum(effective_i)
    # effective_i = 1 if retrieved_i.title in gold_titles
    
    effective_count = 0
    for i in range(min(len(retrieved), k)):
        title = str(retrieved[i].get("title") or retrieved[i].get("doc_title") or "").strip()
        # Normalize title
        # (Simple normalization)
        if title in gold_titles:
            effective_count += 1
            
    return effective_count / k

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    means = []
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    data_np = np.array(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_np, size=n, replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline predictions.jsonl")
    parser.add_argument("--experiment", required=True, help="Path to experiment predictions.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for report")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_data = load_jsonl(Path(args.baseline))
    experiment_data = load_jsonl(Path(args.experiment))
    
    # Align data by ID
    base_map = {d["_id"]: d for d in baseline_data}
    exp_map = {d["_id"]: d for d in experiment_data}
    
    common_ids = set(base_map.keys()) & set(exp_map.keys())
    print(f"Evaluating {len(common_ids)} common examples.")
    
    metrics = {
        "recall@2": [], "recall@5": [], "recall@10": [],
        "ndcg@2": [], "ndcg@5": [], "ndcg@10": [],
        "ie@2": [], "ie@5": [], "ie@10": []
    }
    
    diffs = {k: [] for k in metrics}
    
    results = {"baseline": {k: [] for k in metrics}, "experiment": {k: [] for k in metrics}}
    
    ks = [2, 5, 10]
    
    for qid in common_ids:
        b_item = base_map[qid]
        e_item = exp_map[qid]
        
        answers = b_item.get("references", [])
        if not answers:
             # Fallback to 'answer' field
             ans = b_item.get("answer")
             if isinstance(ans, str):
                 answers = [ans]
             elif isinstance(ans, list):
                 answers = ans
        
        gold_sp = b_item.get("supporting_facts", []) # [[title, sent_id], ...]
        gold_titles = set(sp[0] for sp in gold_sp)
        
        b_ret = b_item.get("retrieved_context", []) or b_item.get("retrieved_context_topk", [])
        e_ret = e_item.get("retrieved_context", []) or e_item.get("retrieved_context_topk", [])
        
        for k in ks:
            # Baseline
            b_r = calculate_recall_at_k(b_ret, answers, k)
            b_n = calculate_ndcg_at_k(b_ret, answers, k)
            b_ie = calculate_ie_at_k(b_ret, gold_titles, k)
            
            # Experiment
            e_r = calculate_recall_at_k(e_ret, answers, k)
            e_n = calculate_ndcg_at_k(e_ret, answers, k)
            e_ie = calculate_ie_at_k(e_ret, gold_titles, k)
            
            results["baseline"][f"recall@{k}"].append(b_r)
            results["baseline"][f"ndcg@{k}"].append(b_n)
            results["baseline"][f"ie@{k}"].append(b_ie)
            
            results["experiment"][f"recall@{k}"].append(e_r)
            results["experiment"][f"ndcg@{k}"].append(e_n)
            results["experiment"][f"ie@{k}"].append(e_ie)
            
            diffs[f"recall@{k}"].append(e_r - b_r)
            diffs[f"ndcg@{k}"].append(e_n - b_n)
            diffs[f"ie@{k}"].append(e_ie - b_ie)

    # Calculate stats
    report_lines = []
    latex_rows = []
    
    for metric in metrics:
        b_vals = results["baseline"][metric]
        e_vals = results["experiment"][metric]
        
        b_mean = np.mean(b_vals)
        e_mean = np.mean(e_vals)
        
        b_ci = bootstrap_ci(b_vals)
        e_ci = bootstrap_ci(e_vals)
        
        # Significance
        p_value = stats.ttest_rel(b_vals, e_vals).pvalue
        sig = "*" if p_value < 0.05 else ""
        if p_value < 0.01: sig = "**"
        if p_value < 0.001: sig = "***"
        
        line = f"{metric:<10} | Base: {b_mean:.4f} ({b_ci[0]:.4f}-{b_ci[1]:.4f}) | Exp: {e_mean:.4f} ({e_ci[0]:.4f}-{e_ci[1]:.4f}) | p={p_value:.4f} {sig}"
        report_lines.append(line)
        
        # LaTeX row: Metric & Base & Exp & p-value
        latex_rows.append(f"{metric.replace('_', ' ').upper()} & {b_mean:.3f} & {e_mean:.3f} & {p_value:.3f} {sig} \\\\")
        
    # Write report
    with (out_dir / "report.txt").open("w") as f:
        f.write("\n".join(report_lines))
        
    with (out_dir / "table.tex").open("w") as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Baseline (Sentence) & Experiment (Fixed) & P-Value \\\\\n")
        f.write("\\midrule\n")
        f.write("\n".join(latex_rows))
        f.write("\n\\bottomrule\n")
        f.write("\\end{tabular}\n")
        
    # Plots
    # Recall Curve
    plt.figure()
    b_recalls = [np.mean(results["baseline"][f"recall@{k}"]) for k in ks]
    e_recalls = [np.mean(results["experiment"][f"recall@{k}"]) for k in ks]
    plt.plot(ks, b_recalls, marker='o', label='Baseline (Sentence)')
    plt.plot(ks, e_recalls, marker='x', label='Experiment (Fixed)')
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.title('Recall@k Curve')
    plt.legend()
    plt.savefig(out_dir / "recall_curve.png")
    
    # NDCG Curve
    plt.figure()
    b_ndcgs = [np.mean(results["baseline"][f"ndcg@{k}"]) for k in ks]
    e_ndcgs = [np.mean(results["experiment"][f"ndcg@{k}"]) for k in ks]
    plt.plot(ks, b_ndcgs, marker='o', label='Baseline (Sentence)')
    plt.plot(ks, e_ndcgs, marker='x', label='Experiment (Fixed)')
    plt.xlabel('k')
    plt.ylabel('NDCG')
    plt.title('NDCG@k Curve')
    plt.legend()
    plt.savefig(out_dir / "ndcg_curve.png")
    
    print(f"Analysis complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
