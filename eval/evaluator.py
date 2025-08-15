from typing import List, Dict, Optional
from loguru import logger
from config import config
from query import QueryProcessor
from llm import LocalLLM

class Evaluator:
    """Simple evaluation module processing queries in batch."""
    def __init__(self, atomic_notes: List[Dict], embeddings=None, llm: Optional[LocalLLM] = None):
        self.processor = QueryProcessor(atomic_notes, embeddings, llm=llm)
        self.batch_size = config.get('eval.batch_size', 16)

    def run(self, queries: List[str]) -> List[Dict[str, any]]:
        results = []
        for q in queries:
            try:
                results.append(self.processor.process(q))
            except Exception as e:
                logger.error(f"Evaluation failed for query '{q}': {e}")
        return results
