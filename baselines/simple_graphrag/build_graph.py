import pickle
import json
import re
import asyncio
from typing import Dict, List, Any
from baselines.simple_graphrag.graph import SimpleGraph
from structrag.llm_client import LLMChatClient
from loguru import logger

class GraphBuilder:
    def __init__(self, llm_client: LLMChatClient):
        self.llm_client = llm_client
        self.graph = SimpleGraph()
        self.chunk_store: Dict[str, str] = {}

    async def build(self, docs: Dict[str, str], chunk_size: int = 500, overlap: int = 50, concurrency: int = 10):
        """
        Build the graph from a dictionary of documents (doc_id -> text).
        """
        logger.info("Starting graph construction...")
        total_docs = len(docs)
        processed = 0
        
        sem = asyncio.Semaphore(concurrency)
        tasks = []

        async def process_doc(doc_id, text):
            nonlocal processed
            chunks = self._chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{i}"
                self.chunk_store[chunk_id] = chunk
                
                async with sem:
                    triplets = await self._extract_triplets(chunk)
                
                for triplet in triplets:
                    if not isinstance(triplet, dict):
                        continue
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

        for doc_id, text in docs.items():
            tasks.append(process_doc(doc_id, text))
        
        await asyncio.gather(*tasks)
        
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

    async def _extract_triplets(self, chunk_text: str) -> List[Dict[str, str]]:
        prompt = f"""
        Extract knowledge triplets from the text below.
        Return ONLY a valid JSON list of objects. Each object must have "subject", "relation", "object".
        
        Example format:
        [
          {{"subject": "Apple", "relation": "founded by", "object": "Steve Jobs"}},
          {{"subject": "Steve Jobs", "relation": "born in", "object": "California"}}
        ]
        
        Text:
        {chunk_text}
        
        Output JSON:
        """
        
        messages = [{"role": "user", "content": prompt}]
        try:
            # Call chat_async instead of chat
            response = await self.llm_client.chat_async(
                messages,
                max_tokens=8192,
                temperature=0.0,
                llm_profile="extract",
            )
            content = response.content
            
            # Debug logging to see what the model is actually outputting
            logger.debug(f"Triplet extraction raw output: {content[:500]}...")

            # 1. Try markdown json block
            match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Try raw list structure
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
            
            # 3. Fallback: Try to find individual objects and wrap them
            # Sometimes models output multiple JSON objects not in a list
            matches = re.findall(r'\{.*?\}', content, re.DOTALL)
            if matches:
                data = []
                for m in matches:
                    try:
                        obj = json.loads(m)
                        if 'subject' in obj and 'relation' in obj and 'object' in obj:
                            data.append(obj)
                    except:
                        pass
                if data:
                    return data

            # 4. Fallback: check for reasoning tags <think>...</think> and strip them
            if "<think>" in content:
                # Remove think blocks
                content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                # Try finding JSON in the cleaned content
                match = re.search(r'```json\s*(\[.*?\])\s*```', content_clean, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                match = re.search(r'\[.*?\]', content_clean, re.DOTALL)
                if match:
                    return json.loads(match.group(0))

        except Exception as e:
            # Log the content that failed to parse
            safe_content = content.replace('\n', ' ')[:200] if 'content' in locals() else "No content"
            logger.warning(f"Failed to extract triplets: {e} | Content start: {safe_content}")
        
        return []
