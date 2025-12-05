import json
import re
import pickle
from typing import List, Set, Dict
from baselines.simple_graphrag.graph import SimpleGraph
from structrag.llm_client import LLMChatClient
from loguru import logger

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question with a short phrase. If the answer is not contained in the context, say "unknown"."""

class GraphRetriever:
    def __init__(self, graph_path: str, chunk_store_path: str, llm_client: LLMChatClient):
        self.graph = SimpleGraph.load(graph_path)
        with open(chunk_store_path, 'rb') as f:
            self.chunk_store = pickle.load(f)
        self.llm_client = llm_client

    def answer(self, question: str) -> str:
        # 1. Extract entities from query
        entities = self._extract_query_entities(question)
        logger.info(f"Extracted entities: {entities}")
        
        # 2. Match nodes in graph
        matched_node_ids = self._match_nodes(entities)
        logger.info(f"Matched nodes: {matched_node_ids}")
        
        # 3. Expand graph (hop 1-2)
        relevant_chunk_ids = self._expand_graph(matched_node_ids, hops=2)
        logger.info(f"Collected {len(relevant_chunk_ids)} relevant chunks")
        
        # 4. Retrieve chunks
        chunks = [self.chunk_store[cid] for cid in relevant_chunk_ids if cid in self.chunk_store]
        # Limit chunks to avoid context overflow
        chunks = chunks[:15]
        
        # 5. Generate answer
        return self._generate_answer(question, chunks)

    def _extract_query_entities(self, question: str) -> List[str]:
        prompt = f"""
        Extract important named entities or noun phrases from the question.
        Output format: JSON list of strings.
        
        Question: {question}
        
        Output JSON:
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(messages, max_tokens=8192, temperature=0.0)
            # response might be object or string depending on client implementation
            # LLMChatClient.chat returns string if configured properly, but let's handle both
            content = response if isinstance(response, str) else response.content
            
            # Handle <think> blocks
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content.startswith("<think>"):
                content = re.sub(r"^<think>.*", "", content, flags=re.DOTALL).strip()
                
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

    def _generate_answer(self, question: str, chunks: List[str]) -> str:
        context_blocks = []
        for i, chunk in enumerate(chunks):
            context_blocks.append(f"[{i+1}] {chunk}")
        context_str = "\n\n".join(context_blocks)

        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat(messages, max_tokens=8192, temperature=0.0)
            content = response if isinstance(response, str) else response.content
            
            # Handle <think> blocks
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content.startswith("<think>"):
                content = re.sub(r"^<think>.*", "", content, flags=re.DOTALL).strip()
                
            try:
                from utils.rag_normalization import normalize_model_answer
                return normalize_model_answer(content)
            except ImportError:
                return content
                
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Insufficient evidence"
