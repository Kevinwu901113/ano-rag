
import sys
import pickle
import re
import json
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from unittest.mock import MagicMock
from collections import defaultdict

# --- Mock missing dependencies ---
sys.modules["tiktoken"] = MagicMock()
if "tiktoken" in sys.modules:
    sys.modules["tiktoken"].__spec__ = MagicMock()

# Add raptor to path
sys.path.insert(0, str(Path("RAPTOR/raptor").resolve()))
from raptor.tree_structures import Node, Tree
# from raptor.tree_retrieval import TreeRetriever, TreeRetrieverConfig 
# It seems the file is named tree_retriever.py but maybe the class is different or import path is wrong?
# Let's inspect the file name from `ls -R`: tree_retriever.py
# But maybe the package structure requires `raptor.tree_retriever`?
# Wait, `from raptor.tree_retrieval` failed. Maybe it should be `from raptor.tree_retriever`?
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig


# --- Mock Embedding Model ---
# Since we cannot easily instantiate the real embedding model (requires API key/server),
# AND we suspect the tree might already have embeddings?
# Actually, TreeRetriever needs an embedding model to embed the QUERY.
# If we don't have the embedding server running, we can't retrieve.

# BUT! The user's environment seems to have `http://127.0.0.1:8001/v1/embeddings`.
# We can try to use the OpenAICompatEmbeddingModel if the server is running.
# If not, we are stuck.
# Let's assume the server IS running because the user just ran the previous steps?
# Or maybe not.
# If the server is not running, we can't embed the query, so we can't retrieve.

# Wait, if we can't run retrieval, we can't "solve" the problem.
# But the user said "Terminal#8-21 ... You solve it".
# This implies I should be able to run it.

# Let's check if the server is running.
import requests
def check_server():
    try:
        requests.get("http://127.0.0.1:8001/health")
        return True
    except:
        return False

# We will use the existing `OpenAICompatEmbeddingModel` from `baseline/runners/run_raptor_qa.py`?
# No, we can just use a simple client.

class SimpleEmbeddingModel:
    def __init__(self):
        self.url = "http://127.0.0.1:8001/v1/embeddings"
        
    def create_embedding(self, text):
        try:
            resp = requests.post(self.url, json={"input": text, "model": "EMB"})
            if resp.status_code == 200:
                return resp.json()['data'][0]['embedding']
        except:
            pass
        # Fallback: random embedding for testing if server down? 
        # No, that would give random results.
        # We must fail if server is down.
        raise Exception("Embedding server unavailable")

# --- Metrics Calculation ---
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

def run_retrieval_and_eval(workspace_dir, qa_path, gold_path, node_map_path):
    # Load Node Map
    print(f"Loading node map from {node_map_path}...")
    with open(node_map_path) as f:
        node_map = json.load(f)

    # Load Gold Data
    print(f"Loading gold data from {gold_path}...")
    gold_data = {}
    with open(gold_path) as f:
        for line in f:
            obj = json.loads(line)
            id_ = obj.get('_id', obj.get('id'))
            gold_data[id_] = obj

    # Load QA Data (Questions)
    print(f"Loading questions from {qa_path}...")
    questions = []
    # Map qdoc_id to Title for comparison with 2Wiki Gold
    qdoc_to_title = defaultdict(dict)
    
    with open(qa_path) as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj)
            for d in obj['docs']:
                qdoc_to_title[obj['id']][d['id']] = d['title']

    # Initialize Embedding Model
    print("Initializing embedding model...")
    embed_model = SimpleEmbeddingModel()

    # Metrics
    metrics = {
        'recall@2': [], 'recall@5': [],
        'ie@2': [], 'ie@5': [],
        'ndcg@2': [], 'ndcg@5': []
    }

    print(f"Starting re-retrieval for {len(questions)} questions...")
    
    pbar = tqdm(questions)
    for q_obj in pbar:
        qid = q_obj['id']
        question = q_obj['question']
        
        if qid not in gold_data:
            continue
            
        # Load Tree
        q_dir = Path(workspace_dir) / qid
        tree_path = q_dir / "tree.pkl"
        if not tree_path.exists():
            continue
            
        with open(tree_path, "rb") as f:
            tree = pickle.load(f)
            
        # Configure Retriever
        # We use a simplified config.
        # Note: Raptor's TreeRetriever usually takes a config object.
        # We need to mock it or create it.
        
        # We can manually implement retrieval to avoid dependency hell
        # Retrieval Logic:
        # 1. Embed Question
        # 2. Start at Root (Layer N)
        # 3. Compute cosine similarity with current layer nodes
        # 4. Select Top-K
        # 5. If Top-K has children, go deeper.
        # OR: Just flat retrieval from ALL nodes?
        # Raptor default is "tree_traversal" or "collapsed"?
        # Log file said: "Using collapsed_tree".
        # This means it flattens the tree and retrieves from all nodes?
        # Let's check `run_raptor_qa.py`. It uses `TreeRetriever`.
        # And `TreeRetriever` logic depends on `selection_mode`.
        # Log says: "Selection Mode: top_k", "Start Layer: None".
        # This usually implies traversing the tree or searching all nodes.
        
        # SIMPLIFICATION:
        # If "collapsed_tree" was used (as per logs), Raptor flattens all nodes into a single index.
        # But here we have `tree.pkl`.
        # We can just iterate ALL nodes, compute cosine similarity, and take top-K.
        # This is equivalent to "collapsed" retrieval if we consider all nodes.
        
        q_emb = np.array(embed_model.create_embedding(question))
        
        node_scores = []
        for nid, node in tree.all_nodes.items():
            # Node should have embedding?
            # Check node.embeddings?
            if hasattr(node, "embeddings") and node.embeddings:
                # Calculate Cosine Sim
                # node.embeddings might be a dict {model_name: emb}
                # Log said model name is "EMB".
                n_emb = np.array(node.embeddings.get("EMB"))
                if n_emb is not None:
                    score = np.dot(q_emb, n_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(n_emb))
                    node_scores.append((nid, score))
        
        # Sort by score
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_nodes = node_scores[:5] # We need up to 5
        
        # Map to Doc IDs
        # We need to resolve what these nodes correspond to.
        # Using our node_map!
        
        retrieved_doc_titles = [] # List of set of titles per node
        # Wait, for Recall/NDCG, we need a list of "Is Relevant"
        
        # Gold Titles
        gold_obj = gold_data[qid]
        gold_titles = set(x[0] for x in gold_obj['supporting_facts'])
        total_relevant = len(gold_titles)
        
        # Check relevance for Top 2 and Top 5
        relevance_vector = []
        
        # We use a Set to track unique *relevant facts found* (for Recall)
        # But for NDCG/IE, we look at the ranked list.
        
        # Correct Logic:
        # For each retrieved node i (from 1 to 5):
        #   Get its mapped doc IDs (qdoc_xxx).
        #   Convert qdoc_xxx to Titles.
        #   Check if ANY of these titles are in gold_titles.
        #   If yes, relevance[i] = 1 (or count of new facts?)
        #   Usually: relevance = 1 if the chunk contains at least one supporting fact.
        
        for nid, score in top_k_nodes:
            # Get mapped qdocs
            qdocs = node_map.get(qid, {}).get(str(nid), [])
            # Convert to titles
            titles = set()
            for qd in qdocs:
                t = qdoc_to_title[qid].get(qd)
                if t: titles.add(t)
            
            # Check overlap with gold
            is_rel = 1 if not titles.isdisjoint(gold_titles) else 0
            relevance_vector.append(is_rel)
            
            # For strict Recall, we accumulate ALL unique relevant titles found
            if is_rel:
                # Which gold titles were found?
                found = titles.intersection(gold_titles)
                # But wait, Recall@K is usually "Fraction of Gold Items Retrieved".
                # If Node 1 has Title A, Node 2 has Title B.
                # Recall@2 = 2 / Total.
                # If Node 1 has Title A and B. Recall@1 = 2 / Total?
                # Usually we flatten the retrieved items.
                pass

        # Calculate Metrics
        # Re-calculating properly
        
        # Accumulated Relevant Items found at K
        found_at_2 = set()
        found_at_5 = set()
        
        rel_vec_2 = []
        rel_vec_5 = []
        
        for k_idx, (nid, score) in enumerate(top_k_nodes):
            qdocs = node_map.get(qid, {}).get(str(nid), [])
            titles = set()
            for qd in qdocs:
                t = qdoc_to_title[qid].get(qd)
                if t: titles.add(t)
            
            # Check overlap
            overlap = titles.intersection(gold_titles)
            is_rel = 1 if overlap else 0
            
            if k_idx < 2:
                found_at_2.update(overlap)
                rel_vec_2.append(is_rel)
            if k_idx < 5:
                found_at_5.update(overlap)
                rel_vec_5.append(is_rel)

        # Recall
        metrics['recall@2'].append(len(found_at_2) / total_relevant if total_relevant else 0)
        metrics['recall@5'].append(len(found_at_5) / total_relevant if total_relevant else 0)
        
        # IE (Precision)
        metrics['ie@2'].append(sum(rel_vec_2) / 2.0)
        metrics['ie@5'].append(sum(rel_vec_5) / 5.0)
        
        # NDCG
        metrics['ndcg@2'].append(ndcg_at_k(rel_vec_2, 2, total_relevant))
        metrics['ndcg@5'].append(ndcg_at_k(rel_vec_5, 5, total_relevant))

    # Print Results
    print("\nCorrected Evaluation Results (Raptor):")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--qa_data", required=True)
    parser.add_argument("--gold_data", required=True)
    parser.add_argument("--node_map", required=True)
    
    args = parser.parse_args()
    run_retrieval_and_eval(args.workspace, args.qa_data, args.gold_data, args.node_map)
