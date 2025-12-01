import json
import csv
import re
import string
import sys
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

def evaluate(dataset_file, prediction_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create a map from query to ground truths
    ground_truths_map = {}
    for item in dataset:
        query = item['query']
        answers = item['answer']
        if isinstance(answers, str):
            answers = [answers]
        ground_truths_map[query] = answers

    # Read predictions
    predictions = {}
    with open(prediction_file, 'r', encoding='utf-8') as f:
        # qa.tsv format: Question \t Answer
        # Some lines might have <think> tags, we should probably strip them for evaluation
        # or assume the answer is after the </think> tag if present.
        # Based on the read output, it seems the answer is mixed with thinking.
        # Let's try to extract the final answer.
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            question, raw_answer = parts
            
            # Extract answer from raw_answer which might contain <think>...</think>
            # Strategy: if </think> exists, take text after it. Else take the whole text.
            if '</think>' in raw_answer:
                answer = raw_answer.split('</think>')[-1].strip()
            else:
                answer = raw_answer.strip()
            
            # Refined extraction:
            # 1. Look for "**Answer:**" or "Answer:" at the end.
            # 2. Look for bold text like "**...**" which often contains the answer.
            # 3. Fallback to the whole text.
            
            extracted_answer = answer
            
            # Pattern 1: **Answer:** ...
            match = re.search(r'\*\*Answer:\*\*\s*(.*)', answer, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
            else:
                # Pattern 2: Answer: ... (without bold)
                match = re.search(r'Answer:\s*(.*)', answer, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_answer = match.group(1).strip()
                else:
                    # Pattern 3: Look for bold text **...**
                    # This is a bit risky if there are multiple bold sections, but let's try extracting the first one
                    # or the one that looks like an answer.
                    # Based on debug output: "John Mayne's occupation was **printer, journalist, and poet**."
                    bold_matches = re.findall(r'\*\*(.*?)\*\*', answer)
                    if bold_matches:
                        # Heuristic: usually the first bold part is the key entity if no explicit "Answer:" section
                        extracted_answer = bold_matches[0]
            
            # Remove trailing period if it's a short phrase
            if len(extracted_answer) < 50 and extracted_answer.endswith('.'):
                extracted_answer = extracted_answer[:-1]

            answer = extracted_answer

            # The model output might be verbose (e.g. "John Mayne's occupation was **printer...").
            # We should try to extract the core entity or short phrase if possible, 
            # but for standard QA evaluation (like SQuAD), usually the F1 score handles overlap.
            # However, if the answer is a full sentence and ground truth is a word, F1 might be low
            # if the sentence is long.
            # Let's print some examples to debug.
            if len(predictions) < 3:
                print(f"DEBUG: Q: {question}")
                print(f"DEBUG: Raw A: {raw_answer[:100]}...")
                print(f"DEBUG: Parsed A: {answer}")
                print("-" * 20)
            
            predictions[question] = answer

    exact_match = 0
    f1 = 0
    total = 0
    missing_predictions = 0
    
    # Some stats on match types
    match_types = Counter()

    for question, ground_truths in ground_truths_map.items():
        if question not in predictions:
            missing_predictions += 1
            continue
        
        total += 1
        prediction = predictions[question]
        
        em = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1_val = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths
        )
        
        exact_match += em
        f1 += f1_val
        
        if em:
            match_types['Exact Match'] += 1
        elif f1_val > 0:
            match_types['Partial Match'] += 1
        else:
            match_types['No Match'] += 1
            # Debug mismatches
            if match_types['No Match'] <= 5:
                print(f"MISMATCH:")
                print(f"  Q: {question}")
                print(f"  Pred: {prediction}")
                print(f"  GT: {ground_truths}")
                print("-" * 20)

    if total == 0:
        print("No matching questions found between dataset and predictions.")
        return

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    print(f"Total questions evaluated: {total}")
    print(f"Missing predictions: {missing_predictions}")
    print(f"Exact Match: {exact_match:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Match Types: {dict(match_types)}")

    # Calculate and print overall accuracy (EM) and F1
    print(f"\nOverall Accuracy (Exact Match): {exact_match:.2f}%")
    print(f"Overall F1 Score: {f1:.2f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python eval_qa.py <dataset_json> <qa_tsv>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    prediction_file = sys.argv[2]
    evaluate(dataset_file, prediction_file)
