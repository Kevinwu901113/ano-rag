
import json
import argparse
import numpy as np
from collections import defaultdict

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

def eval_raptor_corrected(pred_file, gold_file, node_map_file):
    # Load Node Map
    with open(node_map_file) as f:
        node_map = json.load(f) # {qid: {node_id: [doc_ids]}}
        
    # Load gold
    gold_data = {}
    with open(gold_file) as f:
        for line in f:
            obj = json.loads(line)
            id_ = obj.get('_id', obj.get('id'))
            
            # Extract Gold Doc IDs (titles in 2Wiki)
            # In 2Wiki, supporting_facts = [[title, sent_id], ...]
            # The 'context' field has [title, sentences].
            # We need to map titles to doc IDs if possible, OR just use titles if the node map uses titles.
            # WAIT: The node map we generated uses 'qdoc_XXXX'.
            # The QA data (from qa.jsonl) also used 'qdoc_XXXX'.
            # But the 2Wiki gold file (2wiki_dev_sample_500.jsonl) uses TITLES in supporting_facts.
            # We need to bridge this gap.
            # LUCKILY, the 'qa.jsonl' (used to generate node map) likely contains the mapping from qdoc_ID to Title.
            
            # We will rely on the fact that we need to evaluate against the GOLD set.
            # But we only have mapping from Node -> qdoc_ID.
            # We need qdoc_ID -> Title to match 2Wiki gold.
            # OR, we need to know which qdoc_ID corresponds to the gold facts.
            
            gold_data[id_] = obj

    # Load QA Data to map qdoc_id -> Title
    # We need this because 2Wiki gold uses TITLES, but Raptor uses qdoc IDs internally.
    qa_path = "result/baseline/run_20260214_084154/data/2wiki/qa.jsonl"
    qdoc_to_title = defaultdict(dict) # {qid: {qdoc_id: title}}
    with open(qa_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj['id']
            for d in obj['docs']:
                qdoc_to_title[qid][d['id']] = d['title']

    metrics = {
        'recall@2': [], 'recall@5': [],
        'ie@2': [], 'ie@5': [],
        'ndcg@2': [], 'ndcg@5': []
    }
    
    count = 0
    with open(pred_file) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj['id']
            if qid not in gold_data:
                continue
            
            count += 1
            
            # Get Gold Titles
            gold_obj = gold_data[qid]
            gold_titles = set(x[0] for x in gold_obj['supporting_facts'])
            total_relevant = len(gold_titles)
            
            # Get Retrieved Nodes (from ctxs if available, otherwise we might be stuck)
            # PROBLEM: The pred.jsonl file provided by user (lines 1-3) does NOT have 'ctxs'.
            # {"id": "...", "pred": "..."}
            # This means we CANNOT calculate retrieval metrics from pred.jsonl alone if it lacks ctxs.
            
            # However, Raptor usually logs retrieval in a separate file or we need to rely on the fact that
            # we have the tree. But we don't know WHICH nodes were retrieved unless they are in pred.jsonl.
            
            # Let's check if there is a retrieval file.
            # Usually: result/.../retrieved_context_topk.jsonl?
            # Or maybe we can't do it without re-running retrieval?
            
            # WAIT: If pred.jsonl lacks ctxs, we can't compute retrieval metrics.
            # Let's check if there are other files in the directory.
            
            # Assuming we can find the retrieved nodes. 
            # If not, we might need to abort or ask user.
            # But let's assume for a moment we have 'ctxs' or can find them.
            
            # If pred.jsonl has no ctxs, we check 'retrieved_context_topk.jsonl' if it exists.
            
            # DEBUG: Assuming pred.jsonl MIGHT have ctxs in later lines, or we can't do anything.
            # But wait! The run_raptor_qa.py saves ctxs!
            # Let's assume the user provided file is incomplete or I missed it.
            # But tool output said: Content from line 1 to line 1: {"id": "...", "pred": "..."}
            # This confirms NO CTXS in pred.jsonl.
            
            # CRITICAL ISSUE: We cannot evaluate retrieval if we don't know what was retrieved.
            # The ONLY place this info might be is in the log file OR if we re-run retrieval.
            # But re-running is expensive.
            
            # Since I'm an autonomous agent and I know Raptor stores retrieval results in memory during execution
            # but seemingly drops them in the final output unless `retrieval_only` is set.
            # Wait, looking at `run_raptor_qa.py`:
            #   pred = normalize_answer_for_eval(str(ra.answer_question(question) or "").strip())
            #   return pred, [], cost
            # IT RETURNS EMPTY LIST FOR CTXS!!!
            
            # This confirms that the current Raptor run DID NOT save retrieval context in `pred.jsonl`.
            # This is a major flaw in the baseline runner.
            
            # HOWEVER, for the purpose of THIS task, the user asked me to "solve then calculate".
            # "Solve" implies I might need to fix the runner and re-run, OR find a way to get the context.
            # Re-running 500 questions takes too long.
            
            # ALTERNATIVE:
            # We have the TREES. We have the QUESTIONS.
            # We can re-run JUST the retrieval step (not QA) using the EXISTING trees.
            # This should be fast because trees are cached.
            
            # Let's try to simulate retrieval using the tree.
            # But we need the query embedding model.
            
            # Can I write a script to re-retrieve top-k nodes for each question using the existing tree?
            # Yes!
            
            print("ERROR: pred.jsonl does not contain 'ctxs'. Cannot evaluate retrieval.")
            print("You need to re-run Raptor with retrieval capture enabled, or use a script to re-retrieve from existing trees.")
            return {}

    return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--gold_file', required=True)
    parser.add_argument('--node_map', required=True)
    args = parser.parse_args()
    
    # Since we identified that we cannot compute metrics without re-retrieving,
    # I will create a NEW script to do: Re-Retrieve -> Map IDs -> Eval.
    # This script will be `scripts/rerun_raptor_retrieval_and_eval.py`
    print("Please run `scripts/rerun_raptor_retrieval_and_eval.py` instead.")

