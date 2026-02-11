import json
import os
import collections
import string
import re

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

def is_refusal(text):
    refusal_keywords = [
        "insufficient information", "does not contain", "cannot be answered", 
        "no information", "not provided", "sorry", "i don't know",
        "context does not mention", "insufficient evidence"
    ]
    norm_text = text.lower()
    for kw in refusal_keywords:
        if kw in norm_text:
            return True
    return False

def analyze_file(file_path, dataset_name):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    total = 0
    refusals = 0
    high_recall_failures = 0
    high_recall_count = 0
    
    failures = []

    print(f"--- Analyzing {dataset_name} ({os.path.basename(file_path)}) ---")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            prediction = data.get('prediction') or data.get('generated_answer') or ""
            gold_answers = data.get('references') or data.get('gold_answer')
            if not isinstance(gold_answers, list):
                gold_answers = [gold_answers] if gold_answers else []
            
            # Check Refusal
            if is_refusal(prediction):
                refusals += 1
            
            # Check Recall (Approximation from metadata if available, else skip complex calc)
            # We can use 'gold_sp' and 'retrieved_context_raw'
            gold_sp = data.get('gold_sp', [])
            gold_titles = set(item[0] for item in gold_sp)
            
            retrieved_context = data.get('retrieved_context_raw', [])
            retrieved_titles = set()
            for item in retrieved_context[:5]: # Top 5
                t = item.get('title') or item.get('doc_title')
                if t: retrieved_titles.add(t)
            
            # Simple Recall@5 Check
            if not gold_titles:
                continue
                
            hits = len(retrieved_titles & gold_titles)
            recall = hits / len(gold_titles)
            
            if recall == 1.0: # Full Recall
                high_recall_count += 1
                
                # Check F1
                best_f1 = 0.0
                for gold in gold_answers:
                    # Simple F1
                    pred_toks = normalize_answer(prediction).split()
                    gold_toks = normalize_answer(gold).split()
                    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
                    num_same = sum(common.values())
                    if len(pred_toks) > 0 and len(gold_toks) > 0:
                        prec = num_same / len(pred_toks)
                        rec = num_same / len(gold_toks)
                        if prec + rec > 0:
                            f1 = 2 * prec * rec / (prec + rec)
                            best_f1 = max(best_f1, f1)
                
                if best_f1 < 0.1: # Failure
                    high_recall_failures += 1
                    if len(failures) < 3:
                        failures.append({
                            "q": data.get("question"),
                            "pred": prediction,
                            "gold": gold_answers,
                            "is_refusal": is_refusal(prediction),
                            "context_titles": list(retrieved_titles)
                        })

    print(f"Total: {total}")
    print(f"Refusal Rate: {refusals/total:.2%} ({refusals}/{total})")
    if high_recall_count > 0:
        print(f"High Recall (R@5=1.0) Count: {high_recall_count}")
        print(f"Conversion Failure (High Recall but F1<0.1): {high_recall_failures/high_recall_count:.2%} ({high_recall_failures}/{high_recall_count})")
    else:
        print("No High Recall cases found (check data).")
        
    print("\nSample Failures (Full Recall but Wrong Answer):")
    for fail in failures:
        print(f"Q: {fail['q']}")
        print(f"Gold: {fail['gold']}")
        print(f"Pred: {fail['pred']} (Refusal: {fail['is_refusal']})")
        print(f"Context: {fail['context_titles']}")
        print("-" * 20)
    print("\n")

def main():
    files = [
        ("/home/wjk/workplace/nq/ano-rag/result/musique_experiment_10half/pred_dev_dense.jsonl", "MuSiQue (Dense)"),
        ("/home/wjk/workplace/nq/ano-rag/result/experiment_24half/pred_dev_hybrid.jsonl", "HotpotQA (Hybrid)"),
        ("/home/wjk/workplace/nq/ano-rag/result/experiment_2wiki_3half/pred_dev_dense.jsonl", "2Wiki (Dense)")
    ]
    
    for path, name in files:
        analyze_file(path, name)

if __name__ == "__main__":
    main()
