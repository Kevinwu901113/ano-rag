
import os
import json
import glob
import re
import numpy as np
import collections
import string

# --- Configuration ---
DATA_DIR = 'data'
RESULT_DIR = 'result'

# Gold files mapping
GOLD_FILES = {
    '2wiki': os.path.join(DATA_DIR, '2wiki_dev_sample_500.jsonl'),
    'hotpotqa': os.path.join(DATA_DIR, 'hotpot_dev_distractor_500.json'),
    'musique': os.path.join(DATA_DIR, 'musique_ans_v1.0_dev_500.jsonl')
}

# Ranges
RANGES = {
    '2wiki': range(0, 9), # experiment_2wiki, experiment_2wiki_1..8 (handled by regex)
    'hotpotqa': range(19, 30),
    'musique': range(1, 16)
}

# --- Metrics Utils ---

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

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0, 0, 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0, 0, 0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_paragraph_recall(retrieved_tuples, gold_titles, k):
    if not gold_titles: return 0.0
    retrieved_k = retrieved_tuples[:k]
    pred_titles = set(item[0] for item in retrieved_k)
    hits = len(pred_titles & gold_titles)
    return hits / len(gold_titles)

def calculate_ie_at_k(retrieved_tuples, gold_titles, k):
    effective_count = 0
    for i in range(min(k, len(retrieved_tuples))):
        title = retrieved_tuples[i][0]
        if title in gold_titles:
            effective_count += 1
    return effective_count / k

def calculate_paragraph_ndcg(retrieved_tuples, gold_titles, k):
    relevance = []
    seen_titles = set()
    retrieved_k = retrieved_tuples[:k]
    for item in retrieved_k:
        title = item[0]
        if title in gold_titles and title not in seen_titles:
            relevance.append(1)
            seen_titles.add(title)
        else:
            relevance.append(0)
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        
    num_gold = len(gold_titles)
    ideal_k = min(num_gold, k)
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0: return 0.0
    return dcg / idcg

# --- Main Logic ---

def load_gold(dataset):
    path = GOLD_FILES[dataset]
    gold_data = {}
    print(f"Loading gold for {dataset} from {path}...")
    try:
        with open(path, 'r') as f:
            # Check json vs jsonl
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                for item in data:
                    gold_data[item['_id']] = item
            else:
                for line in f:
                    try:
                        item = json.loads(line)
                        # Handle different ID keys
                        if '_id' in item:
                            gold_data[item['_id']] = item
                        elif 'id' in item:
                            gold_data[item['id']] = item
                            
                        if dataset == 'musique' and 'paragraphs' in item:
                            sp = []
                            for p in item['paragraphs']:
                                if p.get('is_supporting'):
                                    sp.append([p['title'], p.get('idx', 0)])
                            item['supporting_facts'] = sp
                            if '_id' in item: gold_data[item['_id']] = item
                            elif 'id' in item: gold_data[item['id']] = item
                            
                    except: pass
    except Exception as e:
        print(f"Error loading gold: {e}")
    return gold_data

def evaluate_pred_file(pred_path, gold_data):
    metrics = {k: [] for k in ['em', 'f1', 'recall@2', 'recall@5', 'ie@2', 'ie@5', 'ndcg@2', 'ndcg@5']}
    
    try:
        with open(pred_path, 'r') as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    qid = pred.get('_id', pred.get('id'))
                    if not qid or qid not in gold_data: continue
                    
                    gold = gold_data[qid]
                    
                    # QA
                    p_ans = pred.get('short_answer', pred.get('answer', ''))
                    g_ans = gold.get('answer', '')
                    
                    em = exact_match_score(p_ans, g_ans)
                    f1, _, _ = f1_score(p_ans, g_ans)
                    
                    metrics['em'].append(float(em))
                    metrics['f1'].append(f1)
                    
                    # Retrieval
                    gold_sp = gold.get('supporting_facts', gold.get('gold_sp', []))
                    gold_titles = set(x[0] for x in gold_sp)
                    
                    pred_sp = pred.get('pred_sp_topk', pred.get('pred_sp', pred.get('sp', [])))
                    # Also check if it's stored in 'metrics' as top_k_hit or similar? 
                    # No, we need the list of titles.
                    
                    # Some files use 'retrieved_context_topk'
                    if not pred_sp and 'retrieved_context_topk' in pred:
                        pred_sp = [[x['title'], 0] for x in pred['retrieved_context_topk']]
                    
                    pred_tuples = [tuple(x) for x in pred_sp]
                    
                    metrics['recall@2'].append(calculate_paragraph_recall(pred_tuples, gold_titles, 2))
                    metrics['recall@5'].append(calculate_paragraph_recall(pred_tuples, gold_titles, 5))
                    metrics['ie@2'].append(calculate_ie_at_k(pred_tuples, gold_titles, 2))
                    metrics['ie@5'].append(calculate_ie_at_k(pred_tuples, gold_titles, 5))
                    metrics['ndcg@2'].append(calculate_paragraph_ndcg(pred_tuples, gold_titles, 2))
                    metrics['ndcg@5'].append(calculate_paragraph_ndcg(pred_tuples, gold_titles, 5))
                    
                except: pass
    except: return None
    
    if not metrics['em']: return None
    
    return {k: np.mean(v) for k, v in metrics.items()}

def identify_dataset_and_range(dir_name):
    # 2Wiki
    if '2wiki' in dir_name:
        return '2wiki'
    
    # Check HotpotQA
    # Handle experiment_19, 20...
    if dir_name.startswith('experiment_') and '2wiki' not in dir_name and 'musique' not in dir_name:
        m = re.match(r'experiment_(\d+)', dir_name)
        if m:
            num = int(m.group(1))
            if num in RANGES['hotpotqa']:
                return 'hotpotqa'
        
        # Also try experiment_20_smoke100
        m = re.match(r'experiment_(\d+)_', dir_name)
        if m:
            num = int(m.group(1))
            if num in RANGES['hotpotqa']:
                return 'hotpotqa'

    # MuSiQue
    if dir_name.startswith('musique_experiment_'):
        m = re.match(r'musique_experiment_(\d+)', dir_name)
        if m:
            num = int(m.group(1))
            if num in RANGES['musique']:
                return 'musique'
    
    # Also handle things like musique_experiment_11half
    if dir_name.startswith('musique_experiment_'):
        m = re.search(r'(\d+)', dir_name.replace('musique_experiment_', ''))
        if m:
             num = int(m.group(1))
             if num in RANGES['musique']:
                 return 'musique'
            
    return None

def identify_model_family(model, reader):
    model = str(model).lower()
    reader = str(reader).lower()
    if 'deepseek' in model or 'gpt' in model or reader == 'openai': return 'DeepSeek'
    if 'qwen' in model or 'vllm' in reader: return 'Qwen'
    return 'Unknown'

def scan_best():
    # Store results: dataset -> family -> list of runs
    candidates = {ds: {'DeepSeek': [], 'Qwen': []} for ds in ['2wiki', 'hotpotqa', 'musique']}
    
    # 1. First pass: quick scan of json summaries to find candidates
    for d in os.listdir(RESULT_DIR):
        dir_path = os.path.join(RESULT_DIR, d)
        if not os.path.isdir(dir_path): continue
        
        dataset = identify_dataset_and_range(d)
        if not dataset: continue
        
        # Look for summary_dev.json or metrics*.json
        # Check summary_dev.json
        summary_path = os.path.join(dir_path, 'summary_dev.json')
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                if 'runs' in data:
                    for reader, retrievers in data['runs'].items():
                        for retriever, metrics in retrievers.items():
                            f1 = metrics.get('f1')
                            if f1 is None: f1 = metrics.get('bleu1', metrics.get('rougeL'))
                            if f1 is None: continue
                            
                            if metrics.get('count', 0) < 50: continue # Skip small tests
                            
                            model = metrics.get('model', 'unknown')
                            family = identify_model_family(model, reader)
                            print(f"DEBUG: Found run {dataset} {d} {reader} {retriever} model={model} fam={family} score={f1}")
                            if family in ['DeepSeek', 'Qwen']:
                                candidates[dataset][family].append({
                                'dir': d,
                                'reader': reader,
                                'retriever': retriever,
                                'f1': metrics['f1'],
                                'pred_file': f"pred_dev_{reader}_{retriever}.jsonl" # guess
                            })
                            # print(f"DEBUG: Appended candidate for {dataset} {family}. Count now: {len(candidates[dataset][family])}")
            except: pass
            
        # Check metrics*.json
        for m_file in glob.glob(os.path.join(dir_path, 'metrics*.json')):
            try:
                fname = os.path.basename(m_file)
                if 'stagewise' in fname: continue
                
                parts = fname.replace('metrics.', '').replace('.json', '').split('.')
                reader = parts[0] if len(parts)>0 else 'unknown'
                retriever = parts[1] if len(parts)>1 else 'unknown'
                
                with open(m_file) as f:
                    metrics = json.load(f)
                
                f1 = metrics.get('f1')
                if f1 is None and 'stats' in metrics: f1 = metrics['stats'].get('f1')
                
                # If F1 missing, try alternatives for 2wiki/hotpot
                if f1 is None:
                    if 'stats' in metrics:
                        f1 = metrics['stats'].get('bleu1', metrics['stats'].get('rougeL'))
                    else:
                        f1 = metrics.get('bleu1', metrics.get('rougeL'))
                
                if f1 is None: continue
                
                model = metrics.get('model', metrics.get('stats', {}).get('model', 'unknown'))
                family = identify_model_family(model, reader)
                print(f"DEBUG: Found metrics {dataset} {d} {reader} {retriever} model={model} fam={family} score={f1}")
                
                if family in ['DeepSeek', 'Qwen']:
                    # Add candidate (duplicates handled later by taking max)
                    # We need to know the pred file path
                    pred_path = metrics.get('output_path')
                    if not pred_path:
                        pred_path = os.path.join(dir_path, f"pred_dev_{reader}_{retriever}.jsonl")
                    else:
                        pred_path = os.path.basename(pred_path) # keep relative
                        
                    candidates[dataset][family].append({
                        'dir': d,
                        'reader': reader,
                        'retriever': retriever,
                        'f1': f1,
                        'pred_file': pred_path
                    })
                    # print(f"DEBUG: Appended candidate for {dataset} {family}. Count now: {len(candidates[dataset][family])}")
            except: pass

    # Debug print candidates
    for ds in ['2wiki', 'hotpotqa', 'musique']:
        for fam in ['DeepSeek', 'Qwen']:
            print(f"DEBUG: {ds} {fam} count: {len(candidates[ds][fam])}")

    # 2. Select BEST F1 for each (Dataset, Family) and compute full metrics
    gold_cache = {}
    
    print("\n=== Final Report ===")
    
    for ds in ['2wiki', 'hotpotqa', 'musique']:
        print(f"\n## Dataset: {ds}")
        # Only load gold if we have candidates
        has_cands = any(candidates[ds][f] for f in ['DeepSeek', 'Qwen'])
        if has_cands and ds not in gold_cache:
            gold_cache[ds] = load_gold(ds)
            
        for fam in ['DeepSeek', 'Qwen']:
            cands = candidates[ds][fam]
            if not cands:
                print(f"- {fam}: No results found.")
                continue
            
            # Find best by F1
            best_run = max(cands, key=lambda x: x['f1'])
            
            # Now compute full metrics
            pred_full_path = os.path.join(RESULT_DIR, best_run['dir'], best_run['pred_file'])
            if not os.path.exists(pred_full_path):
                # Try alternatives
                # Sometimes pred_dev_dense.jsonl instead of pred_dev_vllm_dense.jsonl
                alt_path = os.path.join(RESULT_DIR, best_run['dir'], f"pred_dev_{best_run['retriever']}.jsonl")
                if os.path.exists(alt_path):
                    pred_full_path = alt_path
                else:
                    print(f"- {fam}: Best run found ({best_run['dir']}) but pred file missing: {best_run['pred_file']}")
                    continue
            
            print(f"- {fam} Best Run: {best_run['dir']} ({best_run['reader']}/{best_run['retriever']})")
            print(f"  Calculating metrics from {pred_full_path}...")
            
            full_metrics = evaluate_pred_file(pred_full_path, gold_cache[ds])
            
            if full_metrics:
                print(f"  F1: {full_metrics['f1']:.4f}")
                print(f"  EM: {full_metrics['em']:.4f}")
                print(f"  Recall@2: {full_metrics['recall@2']:.4f}")
                print(f"  Recall@5: {full_metrics['recall@5']:.4f}")
                print(f"  IE@2: {full_metrics['ie@2']:.4f}")
                print(f"  IE@5: {full_metrics['ie@5']:.4f}")
                print(f"  NDCG@2: {full_metrics['ndcg@2']:.4f}")
                print(f"  NDCG@5: {full_metrics['ndcg@5']:.4f}")
            else:
                print("  Failed to calculate metrics.")

scan_best()
