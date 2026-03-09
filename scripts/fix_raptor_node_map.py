
import sys
import pickle
import re
import json
import os
from pathlib import Path
from tqdm import tqdm
from unittest.mock import MagicMock

# --- Mock missing dependencies ---
# Raptor imports tiktoken which might be missing.
# We mock it before importing raptor.
sys.modules["tiktoken"] = MagicMock()
# Also mock it for transformers check
if "tiktoken" in sys.modules:
    sys.modules["tiktoken"].__spec__ = MagicMock()


# Add raptor to path
sys.path.insert(0, str(Path("RAPTOR/raptor").resolve()))
from raptor.tree_structures import Node, Tree

def load_qa_docs(qa_path):
    """Load QA data and index docs by QID."""
    qa_data = {}
    with open(qa_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Store doc texts for matching
            qa_data[item['id']] = {d['id']: d['text'] for d in item['docs']}
    return qa_data

def extract_doc_ids_from_text(text):
    match = re.search(r"### DOC (qdoc_\d+)", text)
    if match:
        return {match.group(1)}
    return set()

def resolve_leaf_node_id(node, candidate_docs):
    """Resolve a leaf node to original doc ID(s)."""
    # 1. Try header extraction
    ids = extract_doc_ids_from_text(node.text)
    if ids:
        return ids
        
    # 2. Try text matching against candidates
    node_text = node.text.strip()
    matches = set()
    for doc_id, doc_text in candidate_docs.items():
        # Check exact match or prefix match (since node might be a chunk)
        if node_text in doc_text:
            matches.add(doc_id)
        elif len(node_text) > 20 and node_text[:50] in doc_text:
             matches.add(doc_id)
            
    return matches

def build_node_map(workspace_dir, qa_path, output_path):
    print(f"Loading QA data from {qa_path}...")
    qa_data = load_qa_docs(qa_path)
    
    node_map = {} # {qid: {node_id: [doc_ids]}}
    
    workspace = Path(workspace_dir)
    q_dirs = [d for d in workspace.iterdir() if d.is_dir()]
    
    print(f"Processing {len(q_dirs)} questions in workspace...")
    
    for q_dir in tqdm(q_dirs):
        qid = q_dir.name
        if qid not in qa_data:
            continue
            
        tree_path = q_dir / "tree.pkl"
        if not tree_path.exists():
            continue
            
        try:
            with open(tree_path, "rb") as f:
                tree = pickle.load(f)
        except Exception as e:
            print(f"Error loading tree for {qid}: {e}")
            continue
            
        candidates = qa_data[qid]
        q_map = {}
        
        # 1. Map Leaf Nodes
        leaf_map = {} # {node_id: set(doc_ids)}
        for nid, node in tree.leaf_nodes.items():
            doc_ids = resolve_leaf_node_id(node, candidates)
            leaf_map[nid] = doc_ids
            q_map[nid] = list(doc_ids)
            
        # 2. Map Abstract Nodes (Recursively)
        # Abstract nodes inherit doc_ids from their children
        
        def get_abstract_ids(node_id, memo):
            if node_id in memo: return memo[node_id]
            
            node = tree.all_nodes[node_id]
            ids = set()
            
            # If leaf, use pre-calculated map
            if node_id in leaf_map:
                return leaf_map[node_id]
            
            # If abstract, recurse on children
            if hasattr(node, "children") and node.children:
                for child_id in node.children:
                    if child_id in tree.all_nodes:
                        ids.update(get_abstract_ids(child_id, memo))
            
            memo[node_id] = ids
            return ids
            
        memo = {}
        for nid in tree.all_nodes:
            if nid not in leaf_map:
                doc_ids = get_abstract_ids(nid, memo)
                q_map[nid] = list(doc_ids)
        
        node_map[qid] = q_map
        
    print(f"Saving node map to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(node_map, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Raptor workspace directory (containing QID folders)")
    parser.add_argument("--qa_data", required=True, help="QA jsonl file containing candidate docs")
    parser.add_argument("--output", required=True, help="Output JSON file for node map")
    
    args = parser.parse_args()
    build_node_map(args.workspace, args.qa_data, args.output)
