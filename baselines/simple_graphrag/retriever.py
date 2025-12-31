import json
import re
import pickle
from typing import Any, Dict, List, Set
from baselines.simple_graphrag.graph import SimpleGraph
from structrag.llm_client import LLMChatClient
from loguru import logger
from utils.answer_cleaner import _strip_reasoning
from utils.context_budget import pack_contexts
from utils.output_protocol import build_final_instruction

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}
{final_instruction}

Answer the question with a short phrase.
If the answer is not contained in the context, say "unknown".
"""

from config.config_loader import config as global_config

import faiss
import numpy as np
from baselines.common.model_clients import get_default_embedding_client

class GraphRetriever:
    def __init__(
        self,
        graph_path: str,
        chunk_store_path: str,
        llm_client: LLMChatClient,
        *,
        context_budget: int | None = None,
        dense_index_path: str | None = None,
    ):
        self.graph = SimpleGraph.load(graph_path)
        self.chunk_to_nodes = self.graph.build_chunk_map()
        with open(chunk_store_path, 'rb') as f:
            self.chunk_store = pickle.load(f)
        self.llm_client = llm_client
        self.last_hits: List[Dict[str, Any]] = []
        
        # Load config for relrag parameters
        cfg = global_config.load_config()
        self.relrag_cfg = cfg.get("relrag", {})
        self.top_k = int(self.relrag_cfg.get("top_k", 15))
        self.context_budget = int(context_budget or 0)
        
        # Initialize Dense Index if provided
        self.dense_index = None
        self.dense_encoder = None
        self.chunk_ids_map = None # Needs to map index ID to chunk ID
        
        if dense_index_path:
            logger.info(f"Loading dense index from {dense_index_path}")
            self.dense_index = faiss.read_index(str(dense_index_path))
            self.dense_encoder = get_default_embedding_client(cfg)
            
            # Load meta for mapping index ID to chunk ID
            meta_path = str(dense_index_path) + ".meta.pkl"
            try:
                with open(meta_path, "rb") as f:
                    self.chunk_ids_map = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load dense index meta from {meta_path}: {e}")
                self.dense_index = None # Disable if meta missing

    def answer(self, question: str) -> str:
        relevant_chunk_ids = set()
        
        if self.dense_index and self.chunk_ids_map:
            logger.info("Using Dense -> Graph Expansion strategy")
            relevant_chunk_ids = self._retrieve_dense_expand(question)
        else:
            logger.info("Using Entity Linking -> Graph Expansion strategy")
            # 1. Extract entities from query
            entities = self._extract_query_entities(question)
            logger.info(f"Extracted entities: {entities}")
            
            # 2. Match nodes in graph
            matched_node_ids = self._match_nodes(entities)
            logger.info(f"Matched nodes: {matched_node_ids}")
            
            # 3. Expand graph (hop 1-2)
            relevant_chunk_ids = self._expand_graph(matched_node_ids, hops=1) # Standard LightRAG: 1 hop

        logger.info(f"Collected {len(relevant_chunk_ids)} relevant chunks")
        
        # 4. Retrieve chunks
        hits: List[Dict[str, Any]] = []
        for cid in relevant_chunk_ids:
            if cid not in self.chunk_store:
                continue
            doc_id = None
            if isinstance(cid, str) and "::" in cid:
                doc_id = cid.split("::", 1)[0]
            hits.append(
                {
                    "doc_id": doc_id,
                    "sent_ids": None,
                    "passage_id": str(cid),
                    "score": None,
                    "text": self.chunk_store[cid],
                }
            )
        # Limit chunks to avoid context overflow
        hits = hits[: self.top_k]
        self.last_hits = [
            {**hit, "rank": idx + 1} for idx, hit in enumerate(hits)
        ]
        annotated_hits = []
        for i, hit in enumerate(hits):
            annotated_hits.append({**hit, "text": f"[{i+1}] {hit['text']}"})
        context_str, _, _ = pack_contexts(annotated_hits, self.context_budget)
        
        # 5. Generate answer
        return self._generate_answer(question, context_str)

    def _retrieve_dense_expand(self, question: str, top_k_seeds: int = 5, hops: int = 1) -> Set[str]:
        # 1. Embed Question
        q_vec = self.dense_encoder.encode([question], normalize_embeddings=True)
        
        # 2. Dense Retrieve Seeds
        scores, indices = self.dense_index.search(q_vec, top_k_seeds)
        
        seed_chunk_ids = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunk_ids_map):
                seed_chunk_ids.append(self.chunk_ids_map[idx])
        
        logger.info(f"Dense retrieved seeds: {len(seed_chunk_ids)}")
        
        # 3. Map Seeds to Nodes
        seed_node_ids = set()
        for cid in seed_chunk_ids:
            # chunk_to_nodes maps chunk_id -> set of node_ids
            if cid in self.chunk_to_nodes:
                seed_node_ids.update(self.chunk_to_nodes[cid])
                
        logger.info(f"Mapped to {len(seed_node_ids)} seed nodes")
        
        # 4. Expand Graph
        # Use existing expand logic
        expanded_chunk_ids = self._expand_graph(list(seed_node_ids), hops=hops)
        
        # 5. Union with seeds
        expanded_chunk_ids.update(seed_chunk_ids)
        
        return expanded_chunk_ids

    def _extract_query_entities(self, question: str) -> List[str]:
        prompt = f"""
        Extract important named entities or noun phrases from the question.
        Output format: JSON list of strings.
        
        Question: {question}
        
        Output JSON:
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(
                messages,
                max_tokens=8192,
                temperature=0.0,
                llm_profile="generate",
            )
            # response might be object or string depending on client implementation
            # LLMChatClient.chat returns string if configured properly, but let's handle both
            content = response if isinstance(response, str) else response.content
            
            # Handle <think> blocks
            content = _strip_reasoning(content)
                
            logger.debug(f"Entity extraction response: {content}")
            # Try matching markdown code block first
            match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Fallback to finding first list-like structure
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            # Fallback 2: If no JSON found, split by comma if it looks like a list
            if "," in content and "[" not in content:
                 parts = [p.strip() for p in content.split(",") if p.strip()]
                 if parts:
                     return parts

        except Exception as e:
            logger.warning(f"Failed to extract query entities: {e}")
        return []

    def _match_nodes(self, entities: List[str], top_k: int = 3) -> List[str]:
        matched_ids = set()
        # Normalize graph keys for faster matching
        graph_keys_lower = {k: k.lower() for k in self.graph.nodes.keys()}
        
        for entity in entities:
            entity_norm = entity.lower().strip()
            if not entity_norm:
                continue
                
            # Exact/substring match
            candidates = []
            
            # 1. Try direct lookup first (fast)
            for original_key, lower_key in graph_keys_lower.items():
                if entity_norm == lower_key:
                    candidates.append(original_key)
                    
            # 2. If no exact match, try substring (slower)
            if not candidates:
                 for original_key, lower_key in graph_keys_lower.items():
                    if entity_norm in lower_key or lower_key in entity_norm:
                        candidates.append(original_key)
            
            # Simple heuristic: prioritize exact matches or shorter matches
            # Sort by length difference to prioritize closer matches
            candidates.sort(key=lambda x: abs(len(x) - len(entity)))
            
            matched_ids.update(candidates[:top_k])
            
        return list(matched_ids)

    def _expand_graph(self, start_node_ids: List[str], hops: int = 2) -> Set[str]:
        visited_nodes = set(start_node_ids)
        current_frontier = set(start_node_ids)
        relevant_chunks = set()
        
        # Collect chunks from start nodes
        for nid in start_node_ids:
            node = self.graph.nodes.get(nid)
            if node:
                relevant_chunks.update(node.chunk_ids)
        
        for _ in range(hops):
            next_frontier = set()
            for nid in current_frontier:
                neighbors = self.graph.get_neighbors(nid)
                for edge in neighbors:
                    # Add chunk from edge
                    relevant_chunks.add(edge.chunk_id)
                    
                    neighbor_id = edge.target_id
                    # Also consider undirected/bidirectional graph traversal if needed?
                    # For now assume directed based on Edge definition.
                    
                    if neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        
                        # Add chunks from neighbor node
                        neighbor_node = self.graph.nodes.get(neighbor_id)
                        if neighbor_node:
                            relevant_chunks.update(neighbor_node.chunk_ids)
                            
            current_frontier = next_frontier
            
        return relevant_chunks

    def _generate_answer(self, question: str, context_str: str) -> str:
        prompt = PROMPT_TEMPLATE.format(
            context=context_str,
            question=question,
            final_instruction=build_final_instruction(),
        )

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat(
                messages,
                max_tokens=8192,
                temperature=0.0,
                llm_profile="extract",
            )
            content = response if isinstance(response, str) else response.content
            return content
                
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Insufficient evidence"
