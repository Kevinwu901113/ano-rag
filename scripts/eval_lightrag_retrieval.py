import json
import argparse
import re
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


def _norm_title(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedup_keep_order(items):
    out = []
    seen = set()
    for item in items:
        key = _norm_title(item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(_norm_title(item))
    return out


def _extract_titles_from_references(pred_text):
    """
    Extract titles from LightRAG output format:
    ### References
    - [1] Title One
    - [2] Title Two (content: ...)
    """
    titles = []
    # Find the References section
    ref_match = re.search(r'### References\n(.*)', pred_text, re.DOTALL)
    if not ref_match:
        return titles

    ref_section = ref_match.group(1)

    # Iterate over lines
    for line in ref_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Match pattern: - [1] Title ...
        # We need to capture the title part before (content: or end of line
        # Regex: - \[\d+\] (.*?)(?:\s\(content:.*|$)
        # Note: titles can contain spaces, punctuation.
        m = re.match(r'- \[\d+\] (.*?)(?:\s*\(content:.*|$)', line)
        if m:
            title = m.group(1).strip()
            # Remove any trailing " (content:" if regex missed it (e.g. strict matching)
            if ' (content:' in title:
                title = title.split(' (content:')[0].strip()
            titles.append(title)

    return titles


def _extract_titles_from_contexts(obj):
    for field in ("retrieved_context_topk", "retrieved_context_raw", "ctxs"):
        contexts = obj.get(field)
        if not isinstance(contexts, list):
            continue
        titles = []
        for item in contexts:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("doc_title") or item.get("source_title")
            title = _norm_title(title)
            if title:
                titles.append(title)
        if titles:
            return titles
    return []


def extract_titles(obj):
    # Prefer structured retrieval outputs if available.
    titles = _extract_titles_from_contexts(obj)
    if titles:
        return _dedup_keep_order(titles)

    # Fallback to legacy parsing from generated references text.
    pred_text = obj.get("pred", "")
    return _dedup_keep_order(_extract_titles_from_references(pred_text))

def eval_file(pred_file, gold_file):
    # Load gold
    gold_data = {} # id -> set of supporting titles
    with open(gold_file) as f:
        for line in f:
            obj = json.loads(line)
            id_ = obj.get('_id', obj.get('id'))

            titles = set()
            if 'paragraphs' in obj:
                 # MuSiQue format
                 for p in obj['paragraphs']:
                     if p['is_supporting']:
                         titles.add(p['title'])
            elif 'supporting_facts' in obj:
                # 2Wiki/Hotpot format: list of [title, sent_id]
                for x in obj['supporting_facts']:
                    titles.add(x[0])
            elif 'docs' in obj:
                # baseline/data/<dataset>/qa.jsonl format
                for doc in obj['docs']:
                    if isinstance(doc, dict) and doc.get('is_supporting') is True:
                        title = _norm_title(doc.get('title'))
                        if title:
                            titles.add(title)
            
            gold_data[id_] = titles

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
            if id_ not in gold_data:
                continue

            count += 1
            retrieved_titles = extract_titles(obj)
            
            gold_titles = gold_data[id_]
            total_relevant = len(gold_titles)
            
            # Create relevance vector
            relevance = [1 if t in gold_titles else 0 for t in retrieved_titles]
            
            for k in [2, 5]:
                # Pad relevance with 0 if fewer retrieved items than k
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
