from typing import List, Dict, Any, Set, Union, Optional, Callable
from collections import defaultdict

def normalize_id(text: str) -> str:
    """Normalize ID for consistent matching."""
    if text is None:
        return ""
    return str(text).strip()

def compute_retrieval_metrics(
    retrieved_items: List[Dict[str, Any]], 
    gold_ids: Set[str], 
    k_list: List[int] = [1, 3, 5, 10],
    id_key: str = "doc_id"
) -> Dict[str, float]:
    """
    Compute standard retrieval metrics for a single query.
    
    Args:
        retrieved_items: List of dicts, must contain `id_key`. Sorted by rank/score.
        gold_ids: Set of canonical gold IDs.
        k_list: List of k thresholds.
        id_key: Key to extract ID from retrieved items (e.g. 'doc_id', 'passage_id').
        
    Returns:
        Dict of metrics (R@k, P@k, Hit@k, MRR, AP).
    """
    metrics = {}
    
    # Extract IDs from retrieved items
    # Ensure we use the same normalization
    retrieved_ids = [normalize_id(item.get(id_key)) for item in retrieved_items]
    
    # Calculate Precision, Recall, Hit at K
    # We track *unique* hits for Recall if canonical ID is Doc ID and multiple chunks are retrieved
    # But for Precision, do we count duplicates?
    # Standard IR: If you retrieve duplicate Docs, usually you filter them first.
    # Here we assume retrieved_items might be chunks.
    # If gold is Doc IDs, we should count *unique* Doc IDs found at K.
    
    hits_at_k = {k: 0 for k in k_list}
    unique_hits_at_k = {k: set() for k in k_list}
    
    num_gold = len(gold_ids)
    
    # MRR and AP variables
    first_hit_rank = 0
    precision_sum = 0.0
    num_valid_hits_for_ap = 0
    seen_hits_for_ap = set()
    
    for rank, rid in enumerate(retrieved_ids, 1):
        is_hit = rid in gold_ids
        
        if is_hit:
            # MRR: First hit (any chunk)
            if first_hit_rank == 0:
                first_hit_rank = rank
            
            # MAP: Standard MAP usually expects unique documents if the task is Doc Retrieval.
            # If we count every chunk as a hit, we inflate MAP.
            # Let's assume we want Document-level metrics.
            if rid not in seen_hits_for_ap:
                num_valid_hits_for_ap += 1
                precision_sum += num_valid_hits_for_ap / rank
                seen_hits_for_ap.add(rid)
            
        for k in k_list:
            if rank <= k:
                if is_hit:
                    hits_at_k[k] += 1
                    unique_hits_at_k[k].add(rid)
    
    # Finalize Metrics
    for k in k_list:
        # Recall: Unique Docs Found / Total Gold Docs
        # Precision: Unique Docs Found / k (Strict? Or Hits / k?)
        # Usually P@k = #Relevant / k. If duplicates are relevant, they count in P@k?
        # But if the user wants "Standard IR", duplicate docs in top K is bad.
        # Let's use unique hits for numerator for both to be safe/strict.
        
        unique_h = len(unique_hits_at_k[k])
        metrics[f"Hit@{k}"] = 1.0 if unique_h > 0 else 0.0
        metrics[f"Precision@{k}"] = unique_h / k
        metrics[f"Recall@{k}"] = unique_h / num_gold if num_gold > 0 else 0.0
        
    metrics["MRR"] = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
    metrics["MAP"] = precision_sum / num_gold if num_gold > 0 else 0.0
    
    return metrics

def aggregate_retrieval_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics over all queries."""
    if not per_query_metrics:
        return {}
        
    keys = per_query_metrics[0].keys()
    agg = {k: 0.0 for k in keys}
    
    for m in per_query_metrics:
        for k in keys:
            agg[k] += m.get(k, 0.0)
            
    n = len(per_query_metrics)
    return {k: v / n for k, v in agg.items()}
