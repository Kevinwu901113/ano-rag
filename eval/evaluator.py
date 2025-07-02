from typing import List, Dict
from loguru import logger
from config import config
from query_processor import QueryProcessor

class Evaluator:
    """Simple evaluation module processing queries in batch."""
    def __init__(self, atomic_notes: List[Dict], embeddings=None):
        self.processor = QueryProcessor(atomic_notes, embeddings)
        self.batch_size = config.get('eval.batch_size', 16)

    def run(self, queries: List[str]) -> List[Dict[str, any]]:
        results = []
        for q in queries:
            try:
                results.append(self.processor.process(q))
            except Exception as e:
                logger.error(f"Evaluation failed for query '{q}': {e}")
        return results
