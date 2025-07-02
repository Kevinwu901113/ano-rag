from typing import List, Dict, Any
from loguru import logger
from config import config

class ContextScheduler:
    """Select top notes based on multiple scores."""
    def __init__(self):
        cs = config.get('context_scheduler', {})
        self.t1 = cs.get('semantic_weight', 0.3)
        self.t2 = cs.get('graph_weight', 0.25)
        self.t3 = cs.get('topic_weight', 0.2)
        self.t4 = cs.get('feedback_weight', 0.15)
        self.t5 = cs.get('redundancy_penalty', 0.1)
        self.top_n = cs.get('top_n_notes', 10)

    def schedule(self, candidate_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        seen_contents = set()
        for note in candidate_notes:
            semantic = note.get('retrieval_info', {}).get('similarity', 0)
            graph_score = note.get('centrality', 0)
            topic_score = 1.0 if note.get('cluster_id') is not None else 0.0
            feedback = note.get('feedback_score', 0)
            redundancy = 1.0 if note.get('content') in seen_contents else 0.0
            score = (self.t1 * semantic +
                     self.t2 * graph_score +
                     self.t3 * topic_score +
                     self.t4 * feedback -
                     self.t5 * redundancy)
            seen_contents.add(note.get('content'))
            note['context_score'] = score
            scored.append(note)
        scored.sort(key=lambda x: x.get('context_score', 0), reverse=True)
        selected = scored[:self.top_n]
        logger.info(f"Context scheduler selected {len(selected)} notes")
        return selected
