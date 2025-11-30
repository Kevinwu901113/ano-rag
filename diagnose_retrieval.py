import json
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append("/home/wjk/workplace/nq/ano-rag")

from baselines.naive_rag.runner import NaiveIndex
import faiss
import numpy as np

def diagnose_retrieval(query, chunks_path, index_path):
    print(f"Diagnosing query: {query}")
    
    # Initialize NaiveIndex
    # Note: config is loaded internally by NaiveIndex if not provided
    naive_index = NaiveIndex(
        index_path=index_path,
        chunks_path=chunks_path
    )
    
    # Manually inspect chunks to find the target
    target_chunks = []
    for chunk_id, meta in naive_index._by_vec.items():
        if "Kathy Saltzman" in meta.get("text", ""):
            target_chunks.append((chunk_id, meta))
    
    if not target_chunks:
        print("Target chunk (Kathy Saltzman) not found in loaded chunks.")
        return

    print(f"Found {len(target_chunks)} target chunks containing 'Kathy Saltzman'.")
    for vec_id, chunk in target_chunks:
        print(f"  Vector ID: {vec_id}")
        print(f"  Chunk ID: {chunk.get('chunk_id')}")
        print(f"  Text preview: {chunk.get('text')[:100]}...")

    # Encode query
    print("Encoding query...")
    q_encoded = naive_index._encoder.encode([query])
    if bool(naive_index.embed_cfg.get("normalize", True)):
        faiss.normalize_L2(q_encoded)
    
    # Retrieve top k
    k = 20
    print(f"Searching top {k}...")
    scores, ids = naive_index._index.search(q_encoded.astype("float32"), k)
    
    print("Top results:")
    found_target = False
    for rank, (score, vec_id) in enumerate(zip(scores[0], ids[0]), start=1):
        meta = naive_index._by_vec.get(int(vec_id))
        doc_title = meta.get("meta", {}).get("doc_title") if meta else "Unknown"
        is_target = any(vec_id == t_vec_id for t_vec_id, _ in target_chunks)
        marker = "*** TARGET ***" if is_target else ""
        if is_target:
            found_target = True
        print(f"  Rank {rank}: Score {score:.4f}, Vector ID {vec_id}, Doc Title: {doc_title} {marker}")

    # If target not found, calculate similarity manually
    if not found_target:
        print("\nTarget not in top k. Calculating manual similarity and checking index consistency...")
        
        for t_vec_id, t_chunk in target_chunks:
            t_text = t_chunk.get("text")
            t_encoded = naive_index._encoder.encode([t_text])
            if bool(naive_index.embed_cfg.get("normalize", True)):
                faiss.normalize_L2(t_encoded)
            
            sim = np.dot(q_encoded, t_encoded.T)[0][0]
            print(f"  Manual similarity with Vector ID {t_vec_id}: {sim:.4f}")

            # Check what is actually in the FAISS index at this ID
            try:
                # IndexFlatIP supports reconstruct
                stored_vec = naive_index._index.reconstruct(int(t_vec_id))
                stored_vec = stored_vec.reshape(1, -1)
                
                # Calculate similarity between stored vector and query
                stored_sim = np.dot(q_encoded, stored_vec.T)[0][0]
                print(f"    Similarity with STORED vector at ID {t_vec_id}: {stored_sim:.4f}")
                
                # Calculate similarity between stored vector and manually encoded vector
                self_sim = np.dot(t_encoded, stored_vec.T)[0][0]
                print(f"    Self-similarity (Stored vs Re-encoded): {self_sim:.4f}")
                
                if self_sim < 0.99:
                    print("    CRITICAL: Stored vector does NOT match re-encoded vector!")
            except Exception as e:
                print(f"    Could not reconstruct vector from index: {e}")

if __name__ == "__main__":
    query = "What is Kathy Saltzman's occupation?"
    chunks_path = "result/mirage_naive/chunks.jsonl"
    index_path = "result/mirage_naive/index.faiss"
    diagnose_retrieval(query, chunks_path, index_path)
