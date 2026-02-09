import json
import sys
import os
import re
from collections import Counter, defaultdict
import numpy as np

def parse_hop_from_id(doc_id):
    if doc_id.startswith('2hop'): return 2
    if doc_id.startswith('3hop'): return 3
    if doc_id.startswith('4hop'): return 4
    return 0

def load_original_data(path):
    print(f"Loading Original Data: {path}...")
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                doc_id = row['id']
                
                # Extract Gold info from original
                supporting_paragraphs = [p for p in row.get('paragraphs', []) if p.get('is_supporting')]
                
                # Raw count (number of supporting paragraphs marked in JSON)
                raw_sp_count = len(supporting_paragraphs)
                
                # Unique Titles (to check if multiple paragraphs share same title)
                raw_sp_titles = set(p['title'] for p in supporting_paragraphs)
                unique_title_count = len(raw_sp_titles)
                
                data[doc_id] = {
                    'answerable': row.get('answerable', False),
                    'hop': parse_hop_from_id(doc_id),
                    'raw_sp_count': raw_sp_count,
                    'unique_title_count': unique_title_count,
                    'raw_titles': raw_sp_titles
                }
            except Exception as e:
                print(f"Error parsing line in original: {e}")
                continue
    return data

def load_prediction_data(path):
    print(f"Loading Prediction Data: {path}...")
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                doc_id = row['id']
                
                # Gold SP in prediction file (usually list of [title, idx] or similar)
                gold_sp = row.get('gold_sp', [])
                
                # Extract retrieved titles for Recall calc
                retrieved_context = row.get('retrieved_context_raw', [])
                retrieved_titles = [item.get('title') or item.get('doc_title') for item in retrieved_context]
                retrieved_titles = [str(t).strip() for t in retrieved_titles if t]
                
                data[doc_id] = {
                    'pred_gold_sp_len': len(gold_sp),
                    'pred_gold_titles': set(str(item[0]).strip() for item in gold_sp if len(item) > 0),
                    'retrieved_titles': retrieved_titles
                }
            except Exception as e:
                print(f"Error parsing line in pred: {e}")
                continue
    return data

def calculate_recall(gold_titles, retrieved_titles, k=10):
    if not gold_titles:
        return 0.0
    top_k = set(retrieved_titles[:k])
    hits = sum(1 for t in gold_titles if t in top_k)
    return hits / len(gold_titles)

def audit_integrity(original_path, pred_path):
    orig_data = load_original_data(original_path)
    pred_data = load_prediction_data(pred_path)
    
    # Metrics containers
    stats = {
        'all': {'recalls': [], 'count': 0},
        'answerable': {'recalls': [], 'count': 0},
        'unanswerable': {'recalls': [], 'count': 0}
    }
    
    # For Hop distribution analysis (Answerable only)
    hop_dist = defaultdict(list) # hop -> list of gold_sp lengths
    
    # For Mapping analysis
    mapping_issues = []
    
    print("\nProcessing matched records...")
    
    for doc_id, p_info in pred_data.items():
        if doc_id not in orig_data:
            continue
            
        o_info = orig_data[doc_id]
        
        # 1. Recall Calculation
        recall = calculate_recall(p_info['pred_gold_titles'], p_info['retrieved_titles'], k=10)
        
        stats['all']['recalls'].append(recall)
        stats['all']['count'] += 1
        
        if o_info['answerable']:
            stats['answerable']['recalls'].append(recall)
            stats['answerable']['count'] += 1
            
            # 2. Hop Distribution (using Original Raw Count to see true label distribution)
            # We use UNIQUE title count because our retrieval is title-based
            hop_dist[o_info['hop']].append({
                'raw_count': o_info['raw_sp_count'],
                'unique_count': o_info['unique_title_count'],
                'pred_count': p_info['pred_gold_sp_len']
            })
            
            # 3. Mapping Check
            # Check if we lost gold items during processing
            # We compare Unique Raw Titles vs Pred Gold Titles
            if o_info['unique_title_count'] != p_info['pred_gold_sp_len']:
                 mapping_issues.append({
                     'id': doc_id,
                     'raw': o_info['unique_title_count'],
                     'pred': p_info['pred_gold_sp_len'],
                     'diff': o_info['unique_title_count'] - p_info['pred_gold_sp_len']
                 })
                 
        else:
            stats['unanswerable']['recalls'].append(recall)
            stats['unanswerable']['count'] += 1

    # --- REPORTING ---
    
    print("\n" + "="*50)
    print("AUDIT REPORT: MuSiQue Gold Standard Integrity")
    print("="*50)
    
    # 1. Answerable vs Unanswerable Impact
    print("\n1. Answerable vs Unanswerable Recall Impact")
    print("-" * 40)
    print(f"{'Subset':<15} | {'Count':<8} | {'Avg Recall@10':<15}")
    print(f"{'All':<15} | {stats['all']['count']:<8} | {np.mean(stats['all']['recalls']):.4f}")
    print(f"{'Answerable':<15} | {stats['answerable']['count']:<8} | {np.mean(stats['answerable']['recalls']):.4f}")
    print(f"{'Unanswerable':<15} | {stats['unanswerable']['count']:<8} | {np.mean(stats['unanswerable']['recalls']):.4f}")
    
    # 2. Hop vs Gold Count Distribution (Answerable Only)
    print("\n2. Gold SP Count Distribution by Hop (Answerable Only)")
    print("-" * 80)
    print(f"{'Hop':<5} | {'Metric':<15} | {'Count=1':<8} | {'Count=2':<8} | {'Count=3':<8} | {'Count=4+':<8} | {'Avg Count'}")
    
    for hop in sorted(hop_dist.keys()):
        items = hop_dist[hop]
        total = len(items)
        if total == 0: continue
        
        # Analyze Raw Unique Titles (The Ground Truth in Input)
        raw_counts = [x['unique_count'] for x in items]
        c = Counter(raw_counts)
        avg = np.mean(raw_counts)
        
        def fmt_pct(n): return f"{n/total:.1%}"
        
        print(f"{hop:<5} | {'Raw(Unique)':<15} | {fmt_pct(c[1]):<8} | {fmt_pct(c[2]):<8} | {fmt_pct(c[3]):<8} | {fmt_pct(sum(c[k] for k in c if k>=4)):<8} | {avg:.2f}")
        
        # Analyze Pred Gold SP (What we actually evaluated on)
        pred_counts = [x['pred_count'] for x in items]
        c_p = Counter(pred_counts)
        avg_p = np.mean(pred_counts)
        print(f"{'':<5} | {'Pred(Mapped)':<15} | {fmt_pct(c_p[1]):<8} | {fmt_pct(c_p[2]):<8} | {fmt_pct(c_p[3]):<8} | {fmt_pct(sum(c_p[k] for k in c_p if k>=4)):<8} | {avg_p:.2f}")
        print("-" * 80)

    # 3. Mapping/Filtering Issues
    print("\n3. Mapping & Filtering Integrity")
    print("-" * 40)
    print(f"Total Answerable Samples Checked: {stats['answerable']['count']}")
    print(f"Samples with Count Mismatch (Raw Unique != Pred Gold): {len(mapping_issues)}")
    
    if mapping_issues:
        print("  -> Potential Mapping/Filtering bugs detected!")
        # Analyze direction of mismatch
        lost = sum(1 for x in mapping_issues if x['diff'] > 0)
        gained = sum(1 for x in mapping_issues if x['diff'] < 0)
        print(f"  -> Gold items LOST in mapping: {lost}")
        print(f"  -> Gold items GAINED in mapping: {gained}")
        
        print("\n  Top 5 Mismatch Examples:")
        for m in mapping_issues[:5]:
            print(f"   ID: {m['id']} | Raw: {m['raw']} -> Pred: {m['pred']}")
    else:
        print("  -> No count mismatches found. Mapping seems consistent with Raw Data.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/audit_musique_gold_integrity.py <orig_file> <pred_file>")
        sys.exit(1)
    audit_integrity(sys.argv[1], sys.argv[2])
