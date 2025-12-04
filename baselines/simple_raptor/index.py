import os
import pickle
import math
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils.embedding_utils import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from baselines.simple_raptor.tree import TreeNode

class SimpleRaptorChunker:
    def __init__(self, target_tokens: int = 512, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc_id: str, text: str) -> List[Dict[str, str]]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = self.target_tokens - self.overlap_tokens
        if step < 1:
            step = 1
            
        # Fix: Ensure we at least create one chunk if text exists but is shorter than target_tokens
        if len(words) <= self.target_tokens:
            chunk_text = " ".join(words)
            chunk_id = f"{doc_id}::chunk_0"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })
            return chunks
            
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.target_tokens]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}::chunk_{len(chunks)}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })
            
            if i + self.target_tokens >= len(words):
                break
                
        return chunks

class SimpleRaptorIndexer:
    def __init__(self, embedding_config: Optional[Dict] = None, llm_config: Optional[Dict] = None):
        self.chunker = SimpleRaptorChunker()
        
        # Initialize embedding encoder
        if embedding_config:
            # Use 'qwen3' as default provider for local models
            # Reuse EmbeddingClient to resolve model path and device correctly
            from retriever.embedding_client import EmbeddingClient
            
            # Merge basic settings
            cfg = embedding_config.copy()
            
            # EmbeddingClient expects 'enabled' to be True to do anything usually,
            # but we just want the encoder. We'll use our newly exposed load_encoder method.
            # We don't need to set 'enabled' or load FAISS indices.
            
            # Instantiate client with config
            client = EmbeddingClient(cfg)
            
            # Force load ONLY the encoder part
            client.load_encoder()
            self.encoder = client._encoder
        else:
            # Fallback to default config via EmbeddingClient
            # IMPORTANT: Do NOT default to Qwen/Qwen2.5-7B-Instruct as it is a large LLM, not an embedding model.
            # Use 'qwen3' provider default or let EmbeddingClient resolve it.
            from retriever.embedding_client import EmbeddingClient
            client = EmbeddingClient({}) # Empty config allows it to pick up defaults (e.g. Qwen/Qwen3-Embedding-8B)
            client.load_encoder()
            self.encoder = client._encoder
        
        # Initialize LLM client for summarization
        if llm_config:
            self.llm = LLMChatClient(
                endpoint=llm_config.get("endpoint"),
                model=llm_config.get("model"),
                temperature=llm_config.get("temperature", 0.0),
                stop=llm_config.get("stop")
            )
        else:
            # Load from global config
            from config.config_loader import config as global_config
            cfg = global_config.load_config()
            
            # Try to find LM Studio config
            lm_cfg = cfg.get("lmstudio", {})
            
            endpoint = lm_cfg.get("endpoint", "http://127.0.0.1:1234/v1")
            model = lm_cfg.get("model", "qwen2.5-7b-instruct")
            temperature = float(lm_cfg.get("temperature", 0.0))
            stop = lm_cfg.get("stop")
            
            self.llm = LLMChatClient(
                endpoint=endpoint,
                model=model,
                temperature=temperature,
                stop=stop
            )

        self.nodes: List[TreeNode] = []
        self.chunk_store: Dict[str, str] = {}
        self.index = None
        self.node_ids_map: Dict[int, int] = {} # map faiss index to node_id
        
        # Raptor config
        self.cluster_size = 16
        self.max_root_nodes = 10
        self.max_descendants = 20
        self.summary_prompt_template = """You are a helpful assistant. Please summarize the following text.
The summary should be concise and capture the main points. Do not add any explanation or extra words.

Text:
{text}

Summary:"""

    def build(self, docs: Dict[str, str], cluster_size: int = 16) -> Dict[str, Any]:
        self.cluster_size = cluster_size
        
        # 1. Chunking
        logger.info(f"Chunking {len(docs)} documents...")
        leaf_chunks = []
        for doc_id, text in tqdm(docs.items(), desc="Chunking"):
            doc_chunks = self.chunker.chunk(doc_id, text)
            leaf_chunks.extend(doc_chunks)
        
        # 2. Create Leaf Nodes and Embed
        logger.info(f"Creating {len(leaf_chunks)} leaf nodes...")
        current_level_nodes = []
        chunk_texts = [c["text"] for c in leaf_chunks]
        
        # Batch encode leaf nodes
        logger.info("Encoding leaf nodes...")
        embeddings = self.encoder.encode(chunk_texts)
        if len(embeddings) > 0:
             faiss.normalize_L2(embeddings)
        
        node_counter = 0
        for i, chunk in enumerate(leaf_chunks):
            node = TreeNode(
                node_id=node_counter,
                level=0,
                text=chunk["text"],
                children=[],
                is_leaf=True,
                descendant_chunk_ids=[chunk["chunk_id"]]
            )
            current_level_nodes.append(node)
            self.nodes.append(node)
            self.chunk_store[chunk["chunk_id"]] = chunk["text"]
            node_counter += 1
            
        # Keep track of embeddings for current level
        current_embeddings = embeddings
        
        # 3. Build Tree Recursively
        level = 0
        while len(current_level_nodes) > self.max_root_nodes:
            logger.info(f"Building level {level+1} from {len(current_level_nodes)} nodes...")
            
            # Cluster
            n_clusters = math.ceil(len(current_level_nodes) / self.cluster_size)
            # Ensure at least 1 cluster and not more clusters than samples
            n_clusters = max(1, min(n_clusters, len(current_level_nodes)))
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(current_embeddings)
            labels = kmeans.labels_
            
            new_level_nodes = []
            new_level_texts = []
            
            # Group nodes by cluster
            clusters: Dict[int, List[TreeNode]] = {i: [] for i in range(n_clusters)}
            for idx, label in enumerate(labels):
                clusters[label].append(current_level_nodes[idx])
            
            # Summarize each cluster
            for label, cluster_nodes in tqdm(clusters.items(), desc=f"Summarizing Level {level+1}"):
                if not cluster_nodes:
                    continue
                
                # Aggregate text
                combined_text = "\n\n".join([n.text for n in cluster_nodes])
                
                # Generate summary
                # Truncate text to avoid context window issues, now that we fixed 400s
                safe_text = combined_text[:15000] # 15k chars ~ 4k tokens, safe for Qwen
                prompt = self.summary_prompt_template.format(text=safe_text) 
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                # LLMChatClient.chat returns string content directly
                summary = self.llm.chat(messages, max_tokens=512)
                
                # Aggregate descendants
                descendants = []
                for n in cluster_nodes:
                    descendants.extend(n.descendant_chunk_ids)
                # Limit descendants to keep size manageable
                descendants = list(set(descendants))[:self.max_descendants]
                
                # Create new node
                new_node = TreeNode(
                    node_id=node_counter,
                    level=level + 1,
                    text=summary,
                    children=[n.node_id for n in cluster_nodes],
                    is_leaf=False,
                    descendant_chunk_ids=descendants
                )
                new_level_nodes.append(new_node)
                new_level_texts.append(summary)
                self.nodes.append(new_node)
                node_counter += 1
            
            if not new_level_texts:
                break
                
            # Encode new level
            logger.info(f"Encoding {len(new_level_texts)} summary nodes...")
            current_embeddings = self.encoder.encode(new_level_texts)
            if len(current_embeddings) > 0:
                faiss.normalize_L2(current_embeddings)
            current_level_nodes = new_level_nodes
            level += 1
            
        # 4. Index All Nodes
        logger.info("Building final index for all nodes...")
        all_texts = [n.text for n in self.nodes]
        
        if not all_texts:
             logger.warning("No nodes to index! Creating an empty index with default dimension.")
             # Use encoder's default dimension if available, or a fallback
             # We can try to encode a dummy text to get dimension
             dummy_emb = self.encoder.encode(["test"])
             d = dummy_emb.shape[1]
             self.index = faiss.IndexFlatIP(d)
             return {
                 "num_nodes": 0,
                 "num_levels": level + 1,
                 "dimension": d
             }

        # Re-encode everything to be safe and uniform (though leaf embeddings exist, summaries need encoding if not stored)
        # We already have embeddings for levels step-by-step, but to simplify final index construction:
        # Let's batch encode all nodes at once to form the final index.
        # Note: This might be redundant but ensures a single consistent index matrix.
        
        final_embeddings = self.encoder.encode(all_texts)
        if len(final_embeddings) > 0:
             faiss.normalize_L2(final_embeddings)
        
        d = final_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(final_embeddings)
        
        # Map index ID to node ID (trivial here since we appended in order, but good for safety)
        self.node_ids_map = {i: n.node_id for i, n in enumerate(self.nodes)}
        
        return {
            "num_nodes": len(self.nodes),
            "num_levels": level + 1,
            "dimension": d
        }

    def save(self, index_path: str, nodes_path: str, chunk_store_path: str) -> None:
        if self.index is None:
            raise ValueError("Index not built yet")
            
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        logger.info(f"Saving nodes to {nodes_path}")
        with open(nodes_path, "wb") as f:
            # Save as list of dicts for better compatibility
            pickle.dump([n.to_dict() for n in self.nodes], f)
            
        node_ids_path = index_path + ".node_ids.pkl"
        logger.info(f"Saving node_ids map to {node_ids_path}")
        with open(node_ids_path, "wb") as f:
            pickle.dump(self.node_ids_map, f)
            
        logger.info(f"Saving chunk store to {chunk_store_path}")
        with open(chunk_store_path, "wb") as f:
            pickle.dump(self.chunk_store, f)
