import argparse
import json
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Any

def get_gold_titles(gold_entry: Dict[str, Any]) -> Set[str]:
    """Extract gold titles from supporting_facts or paragraphs."""
    titles = set()
    
    # Standard format: supporting_facts = [[title, sent_id], ...]
    if 'supporting_facts' in gold_entry:
        for item in gold_entry['supporting_facts']:
            if isinstance(item, list) and len(item) > 0:
                titles.add(item[0])
                
    # MuSiQue format: paragraphs list with is_supporting field
    if 'paragraphs' in gold_entry:
        for p in gold_entry['paragraphs']:
            if p.get('is_supporting', False):
                titles.add(p.get('title', ''))

    # AnoRAG/Baseline format: docs list with is_supporting field
    if 'docs' in gold_entry:
        for p in gold_entry['docs']:
            if p.get('is_supporting', False):
                titles.add(p.get('title', ''))
    
    return titles

def calculate_recall(retrieved_titles: List[str], gold_titles: Set[str]) -> float:
    if not gold_titles:
        return 0.0
    unique_retrieved = set(retrieved_titles)
    hits = len(unique_retrieved & gold_titles)
    return hits / len(gold_titles)

def calculate_recall_from_text(retrieved_texts: List[str], gold_titles: Set[str]) -> float:
    if not gold_titles:
        return 0.0
    
    # Check if any gold title is in any retrieved text
    # This is a loose metric: "Is the document mentioned?"
    # We treat all retrieved texts as a single "retrieved set".
    # Recall = (Num Gold Titles found in Texts) / (Num Gold Titles)
    
    hits = 0
    # Combine all texts for faster search? Or search individually?
    # Gold titles can be short, so be careful of false positives.
    # But for baseline, strict string inclusion is standard for "retrieval via generation".
    
    combined_text = " ".join(retrieved_texts).lower()
    
    for title in gold_titles:
        if title.lower() in combined_text:
            hits += 1
            
    return hits / len(gold_titles)

def calculate_ie(retrieved_titles: List[str], gold_titles: Set[str], k: int) -> float:
    if k == 0: return 0.0
    # IE@K = (Number of relevant items in top K) / K
    # Here we count relevant *titles*. 
    # If multiple chunks from same title are retrieved, should they count?
    # Standard Interpretation: Precision@K.
    # If I retrieve 5 chunks from 1 relevant doc, and K=5.
    # Are 5 chunks relevant? Yes.
    # So we check if title is in gold_titles for each item.
    hits = 0
    for title in retrieved_titles[:k]:
        if title in gold_titles:
            hits += 1
    return hits / k

def calculate_ndcg(retrieved_titles: List[str], gold_titles: Set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    
    dcg = 0.0
    idcg = 0.0
    
    # IDCG: Best possible ordering.
    # We have N relevant docs (len(gold_titles)).
    # In ideal case, first N items are relevant.
    # But wait, if we retrieve Chunks, we might have multiple chunks per doc.
    # If we treat this as "Title Retrieval", we should deduplicate titles in retrieved list?
    # Memory says: "For NDCG... deduplicate by (title, sent_id)... Prevent double-counting".
    # But here we only have Titles from LightRAG (maybe).
    # If LightRAG output has chunk_id, we can't map to sent_id easily without gold.
    # Let's stick to Title-Level NDCG with deduplication (First hit counts).
    
    # DCG
    seen_titles = set()
    for i, title in enumerate(retrieved_titles[:k]):
        rel = 0
        if title in gold_titles and title not in seen_titles:
            rel = 1
            seen_titles.add(title)
        
        dcg += rel / math.log2(i + 2)
        
    # IDCG
    # Ideal: First |Gold| items are relevant (rel=1).
    # Assuming we want to find all |Gold| titles.
    num_gold = len(gold_titles)
    for i in range(min(num_gold, k)):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    args = parser.parse_args()

    # Load Gold
    gold_data = {}
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                qid = str(item.get('_id') or item.get('id'))
                gold_data[qid] = item
            except:
                pass

    # Load Pred
    preds = []
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                preds.append(json.loads(line))
            except:
                pass

    metrics = {
        'recall@2': [], 'recall@5': [],
        'ie@2': [], 'ie@5': [],
        'ndcg@2': [], 'ndcg@5': []
    }

    for pred in preds:
        qid = str(pred.get('_id') or pred.get('id'))
        if qid not in gold_data:
            continue
            
        gold_entry = gold_data[qid]
        gold_titles = get_gold_titles(gold_entry)
        
        # Extract Retrieved Context
        # Try 'retrieved_context', 'retrieved_context_topk', 'ctxs'
        ctxs = pred.get('retrieved_context', pred.get('retrieved_context_topk', pred.get('ctxs', [])))
        
        # Check format
        if ctxs and isinstance(ctxs[0], str):
            # List of strings (GraphRAG/Raptor dump)
            # We can only compute Recall (Context Recall)
            # IE and NDCG are not well-defined for a single text dump without ranking
            # So we just assign Recall value to all @K metrics for simplicity, or 0 for rank-sensitive ones?
            # Let's assign Recall to Recall@K, and 0 to others to indicate "not ranked".
            # Or better: Answer Recall is independent of K if we have 1 chunk.
            
            recall = calculate_recall_from_text(ctxs, gold_titles)
            metrics['recall@2'].append(recall)
            metrics['recall@5'].append(recall)
            # IE/NDCG are 0 or N/A. Let's set 0.
            metrics['ie@2'].append(0.0)
            metrics['ie@5'].append(0.0)
            metrics['ndcg@2'].append(0.0)
            metrics['ndcg@5'].append(0.0)
            continue

        # List of Dicts (Dense/Standard)
        # Extract titles from ctxs
        retrieved_titles = [item.get('title', '') for item in ctxs]
        
        # Calculate Metrics
        metrics['recall@2'].append(calculate_recall(retrieved_titles[:2], gold_titles))
        metrics['recall@5'].append(calculate_recall(retrieved_titles[:5], gold_titles))
        
        metrics['ie@2'].append(calculate_ie(retrieved_titles, gold_titles, 2))
        metrics['ie@5'].append(calculate_ie(retrieved_titles, gold_titles, 5))
        
        metrics['ndcg@2'].append(calculate_ndcg(retrieved_titles, gold_titles, 2))
        metrics['ndcg@5'].append(calculate_ndcg(retrieved_titles, gold_titles, 5))

    # Average
    results = {}
    for k, v in metrics.items():
        results[k] = np.mean(v) if v else 0.0
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
