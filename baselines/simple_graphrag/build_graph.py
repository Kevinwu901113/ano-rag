import pickle
import json
import re
from typing import Dict, List, Any
from baselines.simple_graphrag.graph import SimpleGraph
from structrag.llm_client import LLMChatClient
from loguru import logger

class GraphBuilder:
    def __init__(self, llm_client: LLMChatClient):
        self.llm_client = llm_client
        self.graph = SimpleGraph()
        self.chunk_store: Dict[str, str] = {}

    def build(self, docs: Dict[str, str], chunk_size: int = 500, overlap: int = 50):
        """
        Build the graph from a dictionary of documents (doc_id -> text).
        """
        logger.info("Starting graph construction...")
        total_docs = len(docs)
        processed = 0
        
        for doc_id, text in docs.items():
            chunks = self._chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{i}"
                self.chunk_store[chunk_id] = chunk
                
                triplets = self._extract_triplets(chunk)
                for triplet in triplets:
                    if triplet.get('subject') and triplet.get('relation') and triplet.get('object'):
                        self.graph.add_edge(
                            triplet['subject'],
                            triplet['object'],
                            triplet['relation'],
                            chunk_id
                        )
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{total_docs} documents")
        
        logger.info(f"Graph construction complete. Nodes: {len(self.graph.nodes)}, Chunks: {len(self.chunk_store)}")

    def save(self, graph_path: str, chunk_store_path: str):
        self.graph.save(graph_path)
        with open(chunk_store_path, 'wb') as f:
            pickle.dump(self.chunk_store, f)
        logger.info(f"Graph saved to {graph_path}, Chunk store saved to {chunk_store_path}")

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        if not words:
            return chunks
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def _extract_triplets(self, chunk_text: str) -> List[Dict[str, str]]:
        prompt = f"""
        Extract up to 10 knowledge triplets from the following text.
        Output format must be a JSON list of objects with keys "subject", "relation", "object".
        Ignore non-factual or abstract content.
        
        Text:
        {chunk_text}
        
        Output JSON:
        """
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(messages, max_tokens=512, temperature=0.0)
            content = response.content
            # Robust JSON extraction: non-greedy match
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning(f"Failed to extract triplets: {e}")
        
        return []
