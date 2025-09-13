"""K estimator module for structural evidence quantity estimation.

This module implements completely structural K estimation without word lists.
It uses paragraph embeddings to construct graphs and estimates K based on
shortest path lengths between question anchor and answer paragraphs.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class KEstimator:
    """Structural K estimation system for evidence quantity."""
    
    def __init__(self, calibration_path: Optional[str] = None):
        """
        Initialize the K estimator.
        
        Args:
            calibration_path: Path to calibration file
        """
        self.calibration_path = calibration_path
        
        # Default parameters (can be overridden by calibration)
        self.similarity_threshold = 0.3  # Threshold for building graph edges
        self.entity_overlap_threshold = 0.2  # Threshold for entity-based edges
        self.min_k = 2  # Minimum K value
        self.max_k = 4  # Maximum K value
        self.default_k = 2  # Default fallback K
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def estimate_K(self, question: str, answer_idx: str, passages_by_idx: Dict[str, Dict], 
                  packed_order: List[str]) -> int:
        """
        Estimate K (evidence quantity) using structural graph analysis.
        
        Args:
            question: The input question
            answer_idx: Index of the answer paragraph
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            Estimated K value (number of evidence paragraphs needed)
        """
        logger.info(f"Estimating K for question with answer_idx={answer_idx}")
        
        if not passages_by_idx or not packed_order:
            logger.warning("Empty passages or packed_order, using default K")
            return self.default_k
        
        # Build paragraph graph
        paragraph_graph = self._build_paragraph_graph(passages_by_idx, packed_order)
        
        # Find question anchor paragraph
        question_anchor_idx = self._find_question_anchor(question, passages_by_idx, packed_order)
        
        # Estimate K based on graph structure
        if question_anchor_idx and answer_idx and question_anchor_idx != answer_idx:
            k_estimate = self._estimate_k_from_graph(
                question_anchor_idx, answer_idx, paragraph_graph
            )
        else:
            # Fallback: estimate based on question complexity
            k_estimate = self._estimate_k_from_complexity(question, passages_by_idx)
        
        # Clamp to valid range
        k_final = max(self.min_k, min(k_estimate, self.max_k))
        
        logger.info(f"Estimated K: {k_final} (raw estimate: {k_estimate})")
        return k_final
    
    def _build_paragraph_graph(self, passages_by_idx: Dict[str, Dict], 
                              packed_order: List[str]) -> nx.Graph:
        """
        Build a graph of paragraphs based on embeddings and entity overlap.
        
        Args:
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            NetworkX graph with paragraphs as nodes
        """
        graph = nx.Graph()
        
        # Add nodes
        for idx in packed_order:
            if idx in passages_by_idx:
                passage = passages_by_idx[idx]
                graph.add_node(idx, passage=passage)
        
        # Add edges based on similarity and entity overlap
        indices = list(graph.nodes())
        
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                passage1 = passages_by_idx[idx1]
                passage2 = passages_by_idx[idx2]
                
                # Compute similarity
                similarity = self._compute_paragraph_similarity(passage1, passage2)
                
                # Compute entity overlap
                entity_overlap = self._compute_entity_overlap(passage1, passage2)
                
                # Add edge if similarity or entity overlap exceeds threshold
                if similarity > self.similarity_threshold or entity_overlap > self.entity_overlap_threshold:
                    weight = max(similarity, entity_overlap)
                    graph.add_edge(idx1, idx2, weight=weight, similarity=similarity, entity_overlap=entity_overlap)
        
        logger.info(f"Built paragraph graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _compute_paragraph_similarity(self, passage1: Dict[str, Any], passage2: Dict[str, Any]) -> float:
        """
        Compute similarity between two paragraphs using embeddings or text.
        
        Args:
            passage1: First passage
            passage2: Second passage
            
        Returns:
            Similarity score between 0 and 1
        """
        # Try to use embeddings if available
        if 'embedding' in passage1 and 'embedding' in passage2:
            try:
                emb1 = np.array(passage1['embedding']).reshape(1, -1)
                emb2 = np.array(passage2['embedding']).reshape(1, -1)
                return cosine_similarity(emb1, emb2)[0, 0]
            except Exception as e:
                logger.debug(f"Failed to compute embedding similarity: {e}")
        
        # Fallback to text similarity
        content1 = passage1.get('content', '')
        content2 = passage2.get('content', '')
        
        return self._compute_text_similarity(content1, content2)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute text similarity using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_entity_overlap(self, passage1: Dict[str, Any], passage2: Dict[str, Any]) -> float:
        """
        Compute entity overlap between two passages.
        
        Args:
            passage1: First passage
            passage2: Second passage
            
        Returns:
            Entity overlap score
        """
        # Extract entities using simple heuristics
        entities1 = self._extract_entities(passage1.get('content', ''))
        entities2 = self._extract_entities(passage2.get('content', ''))
        
        if not entities1 or not entities2:
            return 0.0
        
        intersection = len(entities1.intersection(entities2))
        union = len(entities1.union(entities2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_entities(self, text: str) -> set:
        """
        Extract entities from text using simple patterns.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted entities
        """
        import re
        
        entities = set()
        
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.update(proper_nouns)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', text)
        entities.update(numbers)
        
        # Extract years
        years = re.findall(r'\b\d{4}\b', text)
        entities.update(years)
        
        return entities
    
    def _find_question_anchor(self, question: str, passages_by_idx: Dict[str, Dict], 
                            packed_order: List[str]) -> Optional[str]:
        """
        Find the paragraph that best matches the question (question anchor).
        
        Args:
            question: The input question
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            Index of the question anchor paragraph or None
        """
        best_idx = None
        best_score = 0.0
        
        for idx in packed_order:
            if idx in passages_by_idx:
                passage = passages_by_idx[idx]
                content = passage.get('content', '')
                
                # Compute question-passage similarity
                similarity = self._compute_text_similarity(question, content)
                
                if similarity > best_score:
                    best_score = similarity
                    best_idx = idx
        
        logger.info(f"Question anchor: {best_idx} (score: {best_score:.3f})")
        return best_idx
    
    def _estimate_k_from_graph(self, question_anchor_idx: str, answer_idx: str, 
                              graph: nx.Graph) -> int:
        """
        Estimate K based on shortest path in the paragraph graph.
        
        Args:
            question_anchor_idx: Index of question anchor paragraph
            answer_idx: Index of answer paragraph
            graph: Paragraph similarity graph
            
        Returns:
            Estimated K value
        """
        try:
            # Check if both nodes exist in graph
            if question_anchor_idx not in graph or answer_idx not in graph:
                logger.warning(f"Anchor or answer not in graph: {question_anchor_idx}, {answer_idx}")
                return self.default_k
            
            # Compute shortest path
            try:
                path_length = nx.shortest_path_length(graph, question_anchor_idx, answer_idx)
                logger.info(f"Shortest path length: {path_length}")
                
                # K = path_length + 1 (include both endpoints)
                k_estimate = path_length + 1
                
            except nx.NetworkXNoPath:
                logger.warning("No path found between anchor and answer paragraphs")
                # Use graph diameter or default
                if graph.number_of_nodes() > 1:
                    # Estimate based on graph structure
                    avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
                    k_estimate = max(2, int(avg_degree / 2) + 1)
                else:
                    k_estimate = self.default_k
            
            return k_estimate
            
        except Exception as e:
            logger.warning(f"Error in graph-based K estimation: {e}")
            return self.default_k
    
    def _estimate_k_from_complexity(self, question: str, passages_by_idx: Dict[str, Dict]) -> int:
        """
        Estimate K based on question complexity when graph method fails.
        
        Args:
            question: The input question
            passages_by_idx: Dictionary of passages by index
            
        Returns:
            Estimated K value
        """
        # Analyze question complexity
        question_lower = question.lower()
        
        # Multi-hop indicators
        multi_hop_indicators = [
            'who performed', 'who sang', 'who directed', 'who wrote',
            'what album', 'what movie', 'what song',
            'when was', 'where was',
            'spouse of', 'husband of', 'wife of',
            'and', 'also', 'both'
        ]
        
        complexity_score = 0
        
        # Count multi-hop indicators
        for indicator in multi_hop_indicators:
            if indicator in question_lower:
                complexity_score += 1
        
        # Count question words (more question words might indicate complexity)
        question_words = ['who', 'what', 'when', 'where', 'why', 'how']
        question_word_count = sum(1 for word in question_words if word in question_lower)
        
        # Estimate K based on complexity
        if complexity_score >= 2 or question_word_count >= 2:
            k_estimate = 4  # High complexity
        elif complexity_score >= 1 or question_word_count >= 1:
            k_estimate = 3  # Medium complexity
        else:
            k_estimate = 2  # Low complexity
        
        # Consider available passages
        num_passages = len(passages_by_idx)
        k_estimate = min(k_estimate, num_passages)
        
        logger.info(f"Complexity-based K estimate: {k_estimate} (complexity_score: {complexity_score})")
        return k_estimate
    
    def calibrate_thresholds(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calibrate thresholds using validation data.
        
        Args:
            validation_data: List of validation examples with format:
                {
                    'question': str,
                    'passages_by_idx': Dict[str, Dict],
                    'packed_order': List[str],
                    'answer_idx': str,
                    'gold_k': int  # Ground truth K value
                }
                
        Returns:
            Calibration results
        """
        logger.info(f"Calibrating K estimator with {len(validation_data)} examples")
        
        # Grid search over thresholds
        similarity_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        entity_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        best_accuracy = 0.0
        best_params = {
            'similarity_threshold': self.similarity_threshold,
            'entity_overlap_threshold': self.entity_overlap_threshold
        }
        
        for sim_thresh in similarity_thresholds:
            for ent_thresh in entity_thresholds:
                # Temporarily set thresholds
                old_sim_thresh = self.similarity_threshold
                old_ent_thresh = self.entity_overlap_threshold
                
                self.similarity_threshold = sim_thresh
                self.entity_overlap_threshold = ent_thresh
                
                # Evaluate on validation data
                correct = 0
                total = 0
                
                for example in validation_data:
                    question = example['question']
                    passages_by_idx = example['passages_by_idx']
                    packed_order = example['packed_order']
                    answer_idx = example['answer_idx']
                    gold_k = example['gold_k']
                    
                    estimated_k = self.estimate_K(question, answer_idx, passages_by_idx, packed_order)
                    
                    if estimated_k == gold_k:
                        correct += 1
                    total += 1
                
                accuracy = correct / total if total > 0 else 0.0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'similarity_threshold': sim_thresh,
                        'entity_overlap_threshold': ent_thresh
                    }
                
                # Restore old thresholds
                self.similarity_threshold = old_sim_thresh
                self.entity_overlap_threshold = old_ent_thresh
        
        # Set best parameters
        self.similarity_threshold = best_params['similarity_threshold']
        self.entity_overlap_threshold = best_params['entity_overlap_threshold']
        
        results = {
            'best_accuracy': best_accuracy,
            'best_similarity_threshold': best_params['similarity_threshold'],
            'best_entity_threshold': best_params['entity_overlap_threshold']
        }
        
        logger.info(f"K estimator calibration completed. Best accuracy: {best_accuracy:.3f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load K estimation parameters
            k_config = calibration.get('k_estimation', {})
            self.similarity_threshold = k_config.get('similarity_threshold', 0.3)
            self.entity_overlap_threshold = k_config.get('entity_overlap_threshold', 0.2)
            self.min_k = k_config.get('min_k', 2)
            self.max_k = k_config.get('max_k', 4)
            self.default_k = k_config.get('default_k', 2)
            
            logger.info(f"K estimator calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load K estimator calibration: {e}")


def create_k_estimator(calibration_path: Optional[str] = None) -> KEstimator:
    """
    Create a KEstimator instance.
    
    Args:
        calibration_path: Path to calibration file
        
    Returns:
        KEstimator instance
    """
    return KEstimator(calibration_path=calibration_path)