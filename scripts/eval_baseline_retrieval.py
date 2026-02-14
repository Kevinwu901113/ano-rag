import json
import argparse
import numpy as np

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k, ground_truth_count):
    if ground_truth_count == 0:
        return 0.
    dcg_max = dcg_at_k(sorted([1.0] * ground_truth_count, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def eval_file(pred_file, gold_file):
    # Load gold
    gold_counts = {}
    with open(gold_file) as f:
        for line in f:
            obj = json.loads(line)
            id_ = obj.get('_id', obj.get('id'))
            
            # Count supporting facts
            if 'paragraphs' in obj:
                 # MuSiQue format
                 gold_counts[id_] = sum(1 for p in obj['paragraphs'] if p['is_supporting'])
            elif 'supporting_facts' in obj:
                # 2Wiki/Hotpot format: list of [title, sent_id]
                # Count unique items
                gold_counts[id_] = len(set(tuple(x) for x in obj['supporting_facts']))
            else:
                 gold_counts[id_] = 1 # Fallback

    # Load pred
    metrics = {
        'recall@2': [], 'recall@5': [],
        'ie@2': [], 'ie@5': [],
        'ndcg@2': [], 'ndcg@5': []
    }
    
    count = 0
    with open(pred_file) as f:
        for line in f:
            obj = json.loads(line)
            id_ = obj['id']
            if id_ not in gold_counts:
                continue
            
            count += 1
            ctxs = obj.get('ctxs', [])
            # relevance vector based on is_supporting
            relevance = [1 if c.get('is_supporting') else 0 for c in ctxs]
            
            total_relevant = gold_counts[id_]
            
            for k in [2, 5]:
                rel_k = relevance[:k]
                num_relevant_retrieved = sum(rel_k)
                
                # Recall: Retrieved Relevant / Total Relevant
                if total_relevant > 0:
                    metrics[f'recall@{k}'].append(min(1.0, num_relevant_retrieved / total_relevant))
                else:
                    metrics[f'recall@{k}'].append(0.0)
                
                # IE (Precision): Retrieved Relevant / K
                metrics[f'ie@{k}'].append(num_relevant_retrieved / k)
                
                # NDCG
                metrics[f'ndcg@{k}'].append(ndcg_at_k(rel_k, k, total_relevant))

    # Calculate averages
    results = {}
    for k, v in metrics.items():
        results[k] = sum(v) / len(v) if v else 0.0
    
    results['count'] = count
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file')
    parser.add_argument('gold_file')
    args = parser.parse_args()
    
    res = eval_file(args.pred_file, args.gold_file)
    print(json.dumps(res, indent=2))
