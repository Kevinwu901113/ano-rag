from typing import List, Dict, Any, Optional, Set
from loguru import logger
import numpy as np
from collections import defaultdict

from config import config


class EvidenceMerger:
    """Evidence merger for combining and deduplicating retrieval results from multiple sub-questions.
    
    This class merges retrieval results from multiple sub-questions, removes duplicates,
    and applies different merging strategies to produce a final evidence set.
    """
    
    def __init__(self):
        """Initialize the EvidenceMerger."""
        self.merge_strategy = config.get('query.subquestion.merge_strategy', 'weighted')
        logger.info(f"EvidenceMerger initialized with strategy: {self.merge_strategy}")
    
    def merge_evidence(
        self, 
        subquestion_results: List[Dict[str, Any]], 
        original_query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Merge evidence from multiple sub-question retrieval results.
        
        Args:
            subquestion_results: List of retrieval results for each sub-question
                Each item should be: {
                    'sub_question': str,
                    'vector_results': List[Dict],
                    'graph_results': List[Dict]
                }
            original_query: The original query for relevance scoring
            query_embedding: Optional query embedding for similarity calculation
            
        Returns:
            Merged and deduplicated evidence list
        """
        try:
            # Collect all evidence from all sub-questions
            all_evidence = self._collect_all_evidence(subquestion_results)
            
            # Remove duplicates based on note_id and content
            unique_evidence = self._deduplicate_evidence(all_evidence)
            
            # Apply merging strategy
            merged_evidence = self._apply_merge_strategy(
                unique_evidence, 
                subquestion_results, 
                original_query,
                query_embedding
            )
            
            logger.info(
                f"Merged evidence: {len(all_evidence)} total -> "
                f"{len(unique_evidence)} unique -> {len(merged_evidence)} final"
            )
            
            return merged_evidence
            
        except Exception as e:
            logger.error(f"Error merging evidence: {e}")
            # Fallback: return all unique evidence
            all_evidence = self._collect_all_evidence(subquestion_results)
            return self._deduplicate_evidence(all_evidence)
    
    def _collect_all_evidence(self, subquestion_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect all evidence from sub-question results.
        
        Args:
            subquestion_results: List of sub-question retrieval results
            
        Returns:
            List of all evidence items with source information
        """
        all_evidence = []
        
        for i, result in enumerate(subquestion_results):
            sub_question = result.get('sub_question', f'sub_question_{i}')
            
            # Collect vector results
            vector_results = result.get('vector_results', [])
            for note in vector_results:
                evidence_item = self._create_evidence_item(
                    note, 
                    source_type='vector',
                    sub_question=sub_question,
                    sub_question_index=i
                )
                all_evidence.append(evidence_item)
            
            # Collect graph results
            graph_results = result.get('graph_results', [])
            for note in graph_results:
                evidence_item = self._create_evidence_item(
                    note,
                    source_type='graph', 
                    sub_question=sub_question,
                    sub_question_index=i
                )
                all_evidence.append(evidence_item)
        
        return all_evidence
    
    def _create_evidence_item(self, note: Dict[str, Any], source_type: str, sub_question: str, sub_question_index: int) -> Dict[str, Any]:
        """Create an evidence item with source information.
        
        Args:
            note: Original note data
            source_type: 'vector' or 'graph'
            sub_question: The sub-question that retrieved this note
            sub_question_index: Index of the sub-question
            
        Returns:
            Evidence item with source metadata
        """
        evidence_item = note.copy()
        
        # Add source metadata，保留原有的source_info
        source_info = note.get('source_info', {}).copy()
        source_info.setdefault('dataset', '')
        source_info.setdefault('qid', '')
        source_info.update({
            'source_type': source_type,
            'sub_question': sub_question,
            'sub_question_index': sub_question_index,
            'original_score': note.get('score', 0.0)
        })
        evidence_item['source_info'] = source_info
        
        # Initialize aggregation fields
        evidence_item['retrieval_count'] = 1
        evidence_item['source_types'] = {source_type}
        evidence_item['sub_questions'] = {sub_question}
        
        return evidence_item
    
    def _deduplicate_evidence(self, all_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate evidence based on note_id and content.
        
        Args:
            all_evidence: List of all evidence items
            
        Returns:
            List of unique evidence items
        """
        unique_evidence = {}
        
        for evidence in all_evidence:
            # Use note_id as primary key, fallback to content hash
            note_id = evidence.get('note_id')
            if note_id:
                key = f"id_{note_id}"
            else:
                content = evidence.get('content', '')
                key = f"content_{hash(content)}"
            
            if key in unique_evidence:
                # Merge with existing evidence
                existing = unique_evidence[key]
                existing['retrieval_count'] += 1
                existing['source_types'].update(evidence['source_types'])
                existing['sub_questions'].update(evidence['sub_questions'])
                
                # Keep the higher score
                if evidence.get('score', 0) > existing.get('score', 0):
                    existing['score'] = evidence.get('score', 0)
                    existing['source_info'] = evidence['source_info']
            else:
                unique_evidence[key] = evidence
        
        return list(unique_evidence.values())
    
    def _apply_merge_strategy(
        self, 
        unique_evidence: List[Dict[str, Any]], 
        subquestion_results: List[Dict[str, Any]],
        original_query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Apply the configured merge strategy to rank and filter evidence.
        
        Args:
            unique_evidence: List of unique evidence items
            subquestion_results: Original sub-question results for context
            original_query: The original query
            query_embedding: Optional query embedding
            
        Returns:
            Ranked and filtered evidence list
        """
        if self.merge_strategy == 'simple':
            return self._simple_merge(unique_evidence)
        elif self.merge_strategy == 'weighted':
            return self._weighted_merge(unique_evidence, len(subquestion_results))
        elif self.merge_strategy == 'ranked':
            return self._ranked_merge(unique_evidence, original_query, query_embedding)
        else:
            logger.warning(f"Unknown merge strategy: {self.merge_strategy}, using simple merge")
            return self._simple_merge(unique_evidence)
    
    def _simple_merge(self, unique_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple merge: sort by original score.
        
        Args:
            unique_evidence: List of unique evidence items
            
        Returns:
            Evidence sorted by score
        """
        return sorted(unique_evidence, key=lambda x: x.get('score', 0), reverse=True)
    
    def _weighted_merge(self, unique_evidence: List[Dict[str, Any]], num_subquestions: int) -> List[Dict[str, Any]]:
        """Weighted merge: consider retrieval frequency and source diversity.
        
        Args:
            unique_evidence: List of unique evidence items
            num_subquestions: Total number of sub-questions
            
        Returns:
            Evidence sorted by weighted score
        """
        for evidence in unique_evidence:
            original_score = evidence.get('score', 0)
            retrieval_count = evidence.get('retrieval_count', 1)
            source_diversity = len(evidence.get('source_types', set()))
            
            # Calculate weighted score
            frequency_weight = retrieval_count / num_subquestions
            diversity_weight = source_diversity / 2.0  # Max 2 source types (vector, graph)
            
            weighted_score = (
                original_score * 0.6 +  # Original relevance score
                frequency_weight * 0.3 +  # How often it was retrieved
                diversity_weight * 0.1   # Source diversity bonus
            )
            
            evidence['weighted_score'] = weighted_score
        
        return sorted(unique_evidence, key=lambda x: x.get('weighted_score', 0), reverse=True)
    
    def _ranked_merge(self, unique_evidence: List[Dict[str, Any]], original_query: str, query_embedding: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Ranked merge: re-rank based on relevance to original query.
        
        Args:
            unique_evidence: List of unique evidence items
            original_query: The original query
            query_embedding: Optional query embedding for similarity
            
        Returns:
            Evidence re-ranked by relevance to original query
        """
        # For now, use weighted merge as fallback
        # In a full implementation, this would re-compute relevance scores
        # against the original query using embedding similarity
        
        logger.info("Ranked merge not fully implemented, using weighted merge")
        return self._weighted_merge(unique_evidence, 1)
    
    def get_merge_statistics(self, merged_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the merged evidence.
        
        Args:
            merged_evidence: The merged evidence list
            
        Returns:
            Dictionary with merge statistics
        """
        if not merged_evidence:
            return {}
        
        total_count = len(merged_evidence)
        source_type_counts = defaultdict(int)
        retrieval_counts = []
        
        for evidence in merged_evidence:
            source_types = evidence.get('source_types', set())
            for source_type in source_types:
                source_type_counts[source_type] += 1
            
            retrieval_counts.append(evidence.get('retrieval_count', 1))
        
        return {
            'total_evidence': total_count,
            'source_type_distribution': dict(source_type_counts),
            'avg_retrieval_count': np.mean(retrieval_counts) if retrieval_counts else 0,
            'max_retrieval_count': max(retrieval_counts) if retrieval_counts else 0,
            'merge_strategy': self.merge_strategy
        }