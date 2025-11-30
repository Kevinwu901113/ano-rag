import json
import re
import pickle
from typing import List, Set, Dict
from baselines.simple_graphrag.graph import SimpleGraph
from structrag.llm_client import LLMChatClient
from loguru import logger

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
            response = self.llm_client.chat(messages, max_tokens=128, temperature=0.0)
            content = response.content
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning(f"Failed to extract query entities: {e}")
        return []

    def _match_nodes(self, entities: List[str], top_k: int = 3) -> List[str]:
        matched_ids = set()
        for entity in entities:
            entity_norm = entity.lower().strip()
            # Exact/substring match
            candidates = []
            for node_id, node in self.graph.nodes.items():
                if entity_norm in node_id or node_id in entity_norm:
                    candidates.append(node_id)
            
            # Simple heuristic: prioritize exact matches or shorter matches
            candidates.sort(key=lambda x: len(x))
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
        context = "\n\n".join(chunks)
        prompt = f"""
        Question: {question}
        
        Relevant Information:
        {context}
        
        Answer:
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(messages, max_tokens=512, temperature=0.0)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
