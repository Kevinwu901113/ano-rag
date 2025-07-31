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
        if not candidate_notes:
            logger.warning("No candidate notes provided to scheduler")
            return []
        
        # 先进行去重，保留相似度最高的版本
        unique_notes = []
        seen_contents = set()
        seen_note_ids = set()
        
        for note in candidate_notes:
            note_id = note.get('note_id')
            content = note.get('content', '')
            
            # 基于note_id去重
            if note_id and note_id in seen_note_ids:
                continue
            
            # 基于内容去重
            if content in seen_contents:
                continue
            
            unique_notes.append(note)
            if note_id:
                seen_note_ids.add(note_id)
            seen_contents.add(content)
        
        logger.info(f"After content deduplication: {len(unique_notes)} unique notes from {len(candidate_notes)} candidates")
        
        # 对去重后的笔记进行评分
        scored = []
        for note in unique_notes:
            semantic = note.get('retrieval_info', {}).get('similarity', 0)
            graph_score = note.get('centrality', 0)
            topic_score = 1.0 if note.get('cluster_id') is not None else 0.0
            feedback = note.get('feedback_score', 0)
            score = (self.t1 * semantic +
                     self.t2 * graph_score +
                     self.t3 * topic_score +
                     self.t4 * feedback)
            note['context_score'] = score
            scored.append(note)
        
        scored.sort(key=lambda x: x.get('context_score', 0), reverse=True)
        
        # 调试信息：显示前几个笔记的分数
        if scored:
            logger.info(f"Top 3 note scores: {[n.get('context_score', 0) for n in scored[:3]]}")
            logger.info(f"Score breakdown for top note: semantic={scored[0].get('retrieval_info', {}).get('similarity', 0)}, graph={scored[0].get('centrality', 0)}, topic={1.0 if scored[0].get('cluster_id') is not None else 0.0}, feedback={scored[0].get('feedback_score', 0)}")
        
        # 确保至少选择一些笔记，即使分数很低
        min_selection = min(3, len(scored))  # 至少选择3个笔记或所有可用笔记
        selected = scored[:max(self.top_n, min_selection)]
        
        logger.info(f"Context scheduler selected {len(selected)} notes from {len(candidate_notes)} candidates")
        return selected


class MultiHopContextScheduler(ContextScheduler):
    """Scheduler with reasoning path awareness"""

    def schedule_for_multi_hop(
        self,
        candidate_notes: List[Dict[str, Any]],
        reasoning_paths: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidate_notes:
            logger.warning("No candidate notes provided to multi-hop scheduler")
            return []
        
        path_scores = self._calculate_path_scores(candidate_notes, reasoning_paths)

        scored_notes = []
        for note in candidate_notes:
            base_score = self._calculate_base_score(note)
            note_id = note.get("note_id")
            path_score = path_scores.get(note_id, 0)
            completeness = self._calculate_completeness_score(note, reasoning_paths)
            total = 0.3 * base_score + 0.4 * path_score + 0.3 * completeness
            note["multi_hop_score"] = total
            scored_notes.append(note)

        # 调试信息：显示前几个笔记的分数
        if scored_notes:
            scored_notes.sort(key=lambda x: x.get("multi_hop_score", 0), reverse=True)
            logger.info(f"Top 3 multi-hop note scores: {[n.get('multi_hop_score', 0) for n in scored_notes[:3]]}")
            if scored_notes:
                top_note = scored_notes[0]
                base_score = self._calculate_base_score(top_note)
                note_id = top_note.get("note_id")
                path_score = path_scores.get(note_id, 0)
                completeness = self._calculate_completeness_score(top_note, reasoning_paths)
                logger.info(f"Score breakdown for top note: base={base_score}, path={path_score}, completeness={completeness}")

        selected = self._ensure_reasoning_chain_completeness(scored_notes, reasoning_paths)
        
        # 确保至少选择一些笔记，即使分数很低
        if not selected and scored_notes:
            min_selection = min(3, len(scored_notes))
            selected = scored_notes[:min_selection]
            logger.info(f"No notes selected by reasoning chain logic, selecting top {len(selected)} notes")
        
        final_selected = selected[: self.top_n]
        logger.info(f"Multi-hop scheduler selected {len(final_selected)} notes from {len(candidate_notes)} candidates")
        return final_selected

    # Helper methods
    def _calculate_base_score(self, note: Dict[str, Any]) -> float:
        semantic = note.get("retrieval_info", {}).get("similarity", 0)
        graph_score = note.get("centrality", 0)
        topic_score = 1.0 if note.get("cluster_id") is not None else 0.0
        feedback = note.get("feedback_score", 0)
        return (
            self.t1 * semantic
            + self.t2 * graph_score
            + self.t3 * topic_score
            + self.t4 * feedback
        )

    def _calculate_path_scores(
        self, candidate_notes: List[Dict[str, Any]], reasoning_paths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for path in reasoning_paths:
            nodes = path.get("path") or path.get("nodes", [])
            score = path.get("path_score", 0.0)
            for nid in nodes:
                scores[nid] = scores.get(nid, 0.0) + score
        return scores

    def _calculate_completeness_score(
        self, note: Dict[str, Any], reasoning_paths: List[Dict[str, Any]]
    ) -> float:
        total = len(reasoning_paths)
        if total == 0:
            return 0.0
        nid = note.get("note_id")
        count = sum(1 for p in reasoning_paths if nid in (p.get("path") or p.get("nodes", [])))
        return count / total

    def _ensure_reasoning_chain_completeness(
        self, notes: List[Dict[str, Any]], paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        selected = []
        covered = set()
        notes.sort(key=lambda x: x.get("multi_hop_score", 0), reverse=True)
        for note in notes:
            nid = note.get("note_id")
            relevant = [p for p in paths if nid in (p.get("path") or p.get("nodes", []))]
            if relevant:
                new_paths = [p for p in relevant if tuple(p.get("path") or p.get("nodes", [])) not in covered]
                if new_paths or len(selected) < 3:
                    selected.append(note)
                    for p in relevant:
                        covered.add(tuple(p.get("path") or p.get("nodes", [])))
            if len(selected) >= self.top_n:
                break
        return selected
