import pickle
import json
import re
from typing import Dict, List, Any
from baselines.simple_graphrag.graph import SimpleGraph
from baselines.direct_llm.runner import DirectLLMClient
from loguru import logger

class GraphBuilder:
    def __init__(self, llm_client: DirectLLMClient):
        self.llm_client = llm_client
        self.graph = SimpleGraph()
        self.chunk_store: Dict[str, str] = {}

    def build(self, docs: Dict[str, str], chunk_size: int = 500, overlap: int = 50):
        """
        Build the graph from a dictionary of documents (doc_id -> text).
        """
        logger.info("Starting graph construction...")
        
        for doc_id, text in docs.items():
            chunks = self._chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{i}"
                self.chunk_store[chunk_id] = chunk
                
                triplets = self._extract_triplets(chunk)
                for triplet in triplets:
                    self.graph.add_edge(
                        triplet['subject'],
                        triplet['object'],
                        triplet['relation'],
                        chunk_id
                    )
        
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
        
        try:
            # We use a simplified approach here, relying on the LLM client from direct_llm baseline
            # We create a temporary message list structure if the client expects it, 
            # but DirectLLMClient seems to be designed for direct prompt input or simple chat.
            # Looking at DirectLLMClient usage in runner.py, it doesn't seem to have a 'chat' method but 
            # likely relies on an internal call. Let's check DirectLLMClient implementation again.
            # Wait, I saw DirectLLMClient in search results, it doesn't have a 'chat' method exposed directly 
            # in the snippet I saw, but it is used in runner.py.
            # Actually, let's look at how DirectLLMClient is implemented in baselines/direct_llm/runner.py
            # It seems I missed the 'chat' or 'generate' method in the search result snippet.
            # I will assume it has a method to generate text. 
            # Let's use a standard requests call if needed or adapt based on available client.
            # Re-checking the search result for baselines/direct_llm/runner.py...
            # It has __init__. It doesn't show the generation method in the snippet.
            # However, structrag/llm_client.py has LLMChatClient.chat.
            # The prompt says "Use Python + existing LLM client".
            # I will use LLMChatClient from structrag/llm_client.py as it seems more robust and standard in this repo.
            pass
        except Exception:
            pass

        # Let's actually switch to using LLMChatClient as it is more standard.
        # But wait, the user prompt said "Use Python + existing LLM client".
        # I'll assume the DirectLLMClient or LLMChatClient is what I should use.
        # Given the codebase, LLMChatClient in structrag/llm_client.py seems perfect.
        
        return []

# Redefine using LLMChatClient for better compatibility
from structrag.llm_client import LLMChatClient

class GraphBuilder:
    def __init__(self, llm_client: LLMChatClient):
        self.llm_client = llm_client
        self.graph = SimpleGraph()
        self.chunk_store: Dict[str, str] = {}

    def build(self, docs: Dict[str, str], chunk_size: int = 500, overlap: int = 50):
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
            # Simple JSON extraction
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to extract triplets: {e}")
        
        return []
