import json
import csv
import sys
import re
import string
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def load_embedding_model():
    """Load sentence-transformers model for embedding similarity."""
    model_name = 'all-MiniLM-L6-v2'  # Using a standard efficient model
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None

def embedding_similarity_score(model, prediction, ground_truths):
    """Calculate max cosine similarity between prediction and ground truths."""
    if not prediction or not ground_truths:
        return 0.0
    
    pred_emb = model.encode([prediction])
    gt_embs = model.encode(ground_truths)
    
    similarities = cosine_similarity(pred_emb, gt_embs)[0]
    return float(np.max(similarities))

def evaluate(dataset_path, results_path):
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create a map for quick lookup: query -> valid answers
    ground_truth_map = {}
    for item in dataset:
        # We normalize the query key to ensure matching, but keep raw answers
        query_norm = normalize_answer(item['query'])
        ground_truth_map[query_norm] = item['answer']
        
    logger.info(f"Loaded {len(ground_truth_map)} queries from dataset.")

    # Load results
    logger.info(f"Loading results from {results_path}...")
    results = []
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    results.append((row[0], row[1]))
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    logger.info(f"Loaded {len(results)} results.")

    # Load embedding model
    emb_model = load_embedding_model()
    
    total_count = 0
    exact_match_total = 0
    f1_total = 0
    embedding_sim_total = 0
    missing_in_dataset = 0

    for query_raw, model_output in results:
        query_norm = normalize_answer(query_raw)
        
        if query_norm not in ground_truth_map:
            missing_in_dataset += 1
            continue
            
        valid_answers = ground_truth_map[query_norm]
        
        # Calculate metrics
        em = metric_max_over_ground_truths(exact_match_score, model_output, valid_answers)
        f1 = metric_max_over_ground_truths(f1_score, model_output, valid_answers)
        
        emb_sim = 0.0
        if emb_model:
            emb_sim = embedding_similarity_score(emb_model, model_output, valid_answers)
        
        exact_match_total += em
        f1_total += f1
        embedding_sim_total += emb_sim
        total_count += 1

        if total_count % 100 == 0:
            logger.info(f"Processed {total_count} queries...")

    if total_count == 0:
        logger.warning("No matching queries found between dataset and results.")
        return

    em_score = 100.0 * exact_match_total / total_count
    f1_score_avg = 100.0 * f1_total / total_count
    emb_score_avg = 100.0 * embedding_sim_total / total_count

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Queries Evaluated: {total_count}")
    print(f"Missing in Dataset: {missing_in_dataset}")
    print("-" * 30)
    print(f"Exact Match (EM): {em_score:.2f}%")
    print(f"F1 Score:         {f1_score_avg:.2f}%")
    if emb_model:
        print(f"Embedding Sim:    {emb_score_avg:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset_path> <results_path>")
        sys.exit(1)
    
    evaluate(sys.argv[1], sys.argv[2])
