import json

def load_jsonl(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data[item['id']] = item
    return data

dense_path = '/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154/pred/dense/musique/deepseek/pred.jsonl'
raptor_path = '/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154/pred/raptor/musique/deepseek/pred.jsonl'
gold_path = '/home/wjk/workplace/nq/ano-rag/result/baseline/run_20260214_084154/data/musique/qa.jsonl'

dense_data = load_jsonl(dense_path)
raptor_data = load_jsonl(raptor_path)
gold_data = load_jsonl(gold_path)

count = 0
for qid, gold in gold_data.items():
    if qid not in dense_data or qid not in raptor_data:
        continue
    
    d_item = dense_data[qid]
    r_item = raptor_data[qid]
    
    # Check metrics if available, otherwise check exact match or string containment
    # Assuming 'f1' or 'em' field might be in pred.jsonl if evaluated, but usually it's just 'prediction'.
    # Let's rely on simple string check if metrics aren't there.
    
    d_pred = d_item.get('pred', '')
    r_pred = r_item.get('pred', '')
    g_ans = gold.get('answer', '')
    
    # We want Dense to fail (e.g., "Insufficient evidence" or not containing answer)
    # And Raptor to succeed (containing answer)
    
    dense_fail = False
    if "insufficient evidence" in d_pred.lower():
        dense_fail = True
    elif g_ans.lower() not in d_pred.lower() and len(d_pred) < 50: # Short wrong answer
        dense_fail = True
        
    raptor_success = False
    if g_ans.lower() in r_pred.lower():
        raptor_success = True
        
    if dense_fail and raptor_success:
        count += 1
        print("-" * 40)
        print(f"Case ID: {qid}")
        print(f"Question: {gold['question']}")
        print(f"Gold Answer: {g_ans}")
        print(f"Dense Pred: {d_pred}")
        print(f"Raptor Pred: {r_pred}")
        
        # Get supporting docs titles
        supporting_titles = [doc['title'] for doc in gold['docs'] if doc['is_supporting']]
        print(f"Supporting Docs: {supporting_titles}")
        
        if count >= 3:
            break
