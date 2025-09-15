"""Dynamic threshold adjustment based on atomic note quality."""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np
from collections import defaultdict
import statistics


class DynamicThresholdAdjuster:
    """Dynamically adjust retrieval and ranking thresholds based on atomic note quality."""
    
    def __init__(self, atomic_notes: List[Dict[str, Any]]):
        """Initialize the dynamic threshold adjuster.
        
        Args:
            atomic_notes: List of atomic notes with quality metrics
        """
        self.atomic_notes = atomic_notes
        self.quality_stats = self._analyze_quality_distribution()
        self.threshold_cache = {}
        
        logger.info(f"Initialized DynamicThresholdAdjuster with {len(atomic_notes)} notes")
        logger.info(f"Quality stats: {self.quality_stats}")
    
    def _analyze_quality_distribution(self) -> Dict[str, float]:
        """Analyze the quality distribution of atomic notes.
        
        Returns:
            Dictionary containing quality statistics
        """
        quality_scores = []
        entity_counts = []
        relation_counts = []
        content_lengths = []
        
        for note in self.atomic_notes:
            # Extract quality indicators
            entities = note.get('entities', [])
            relations = note.get('relations', [])
            content = note.get('content', '')
            
            # Calculate quality score based on multiple factors
            entity_score = min(len(entities) / 5.0, 1.0)  # Normalize to [0,1]
            relation_score = min(len(relations) / 3.0, 1.0)
            content_score = min(len(content) / 500.0, 1.0)
            
            quality_score = (entity_score + relation_score + content_score) / 3.0
            quality_scores.append(quality_score)
            
            entity_counts.append(len(entities))
            relation_counts.append(len(relations))
            content_lengths.append(len(content))
        
        return {
            'mean_quality': statistics.mean(quality_scores) if quality_scores else 0.0,
            'median_quality': statistics.median(quality_scores) if quality_scores else 0.0,
            'std_quality': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
            'min_quality': min(quality_scores) if quality_scores else 0.0,
            'max_quality': max(quality_scores) if quality_scores else 0.0,
            'mean_entities': statistics.mean(entity_counts) if entity_counts else 0.0,
            'mean_relations': statistics.mean(relation_counts) if relation_counts else 0.0,
            'mean_content_length': statistics.mean(content_lengths) if content_lengths else 0.0
        }
    
    def get_adaptive_thresholds(self, 
                               query: str,
                               candidate_notes: List[Dict[str, Any]],
                               base_similarity_threshold: float = 0.7,
                               base_path_score_threshold: float = 0.5) -> Dict[str, float]:
        """Get adaptive thresholds based on query and candidate quality.
        
        Args:
            query: The search query
            candidate_notes: List of candidate notes
            base_similarity_threshold: Base similarity threshold
            base_path_score_threshold: Base path score threshold
            
        Returns:
            Dictionary of adjusted thresholds
        """
        cache_key = f"{hash(query)}_{len(candidate_notes)}_{base_similarity_threshold}_{base_path_score_threshold}"
        
        if cache_key in self.threshold_cache:
            return self.threshold_cache[cache_key]
        
        # Analyze candidate quality
        candidate_quality = self._analyze_candidate_quality(candidate_notes)
        
        # Adjust thresholds based on overall quality
        quality_factor = self._calculate_quality_factor(candidate_quality)
        
        # Adjust similarity threshold
        adjusted_similarity = self._adjust_similarity_threshold(
            base_similarity_threshold, quality_factor, candidate_quality
        )
        
        # Adjust path score threshold
        adjusted_path_score = self._adjust_path_score_threshold(
            base_path_score_threshold, quality_factor, candidate_quality
        )
        
        # Adjust ranking parameters
        ranking_params = self._adjust_ranking_parameters(quality_factor, candidate_quality)
        
        thresholds = {
            'similarity_threshold': adjusted_similarity,
            'path_score_threshold': adjusted_path_score,
            'entity_weight': ranking_params['entity_weight'],
            'relation_weight': ranking_params['relation_weight'],
            'temporal_weight': ranking_params['temporal_weight'],
            'coherence_weight': ranking_params['coherence_weight'],
            'diversity_penalty': ranking_params['diversity_penalty']
        }
        
        self.threshold_cache[cache_key] = thresholds
        
        logger.debug(f"Adaptive thresholds for query '{query[:50]}...': {thresholds}")
        
        return thresholds
    
    def _analyze_candidate_quality(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze the quality of candidate notes.
        
        Args:
            candidates: List of candidate notes
            
        Returns:
            Dictionary containing candidate quality metrics
        """
        if not candidates:
            return {
                'mean_quality': 0.0,
                'quality_variance': 0.0,
                'entity_density': 0.0,
                'relation_density': 0.0,
                'content_richness': 0.0
            }
        
        quality_scores = []
        entity_counts = []
        relation_counts = []
        content_lengths = []
        
        for candidate in candidates:
            entities = candidate.get('entities', [])
            relations = candidate.get('relations', [])
            content = candidate.get('content', '')
            
            # Calculate quality metrics
            entity_score = min(len(entities) / 5.0, 1.0)
            relation_score = min(len(relations) / 3.0, 1.0)
            content_score = min(len(content) / 500.0, 1.0)
            
            quality_score = (entity_score + relation_score + content_score) / 3.0
            quality_scores.append(quality_score)
            
            entity_counts.append(len(entities))
            relation_counts.append(len(relations))
            content_lengths.append(len(content))
        
        return {
            'mean_quality': statistics.mean(quality_scores),
            'quality_variance': statistics.variance(quality_scores) if len(quality_scores) > 1 else 0.0,
            'entity_density': statistics.mean(entity_counts),
            'relation_density': statistics.mean(relation_counts),
            'content_richness': statistics.mean(content_lengths)
        }
    
    def _calculate_quality_factor(self, candidate_quality: Dict[str, float]) -> float:
        """Calculate overall quality factor for threshold adjustment.
        
        Args:
            candidate_quality: Candidate quality metrics
            
        Returns:
            Quality factor between 0.5 and 1.5
        """
        # Compare candidate quality to global quality
        global_quality = self.quality_stats['mean_quality']
        candidate_mean_quality = candidate_quality['mean_quality']
        
        if global_quality == 0:
            return 1.0
        
        # Calculate relative quality
        relative_quality = candidate_mean_quality / global_quality
        
        # Normalize to reasonable range [0.5, 1.5]
        quality_factor = max(0.5, min(1.5, relative_quality))
        
        return quality_factor
    
    def _adjust_similarity_threshold(self, 
                                   base_threshold: float, 
                                   quality_factor: float,
                                   candidate_quality: Dict[str, float]) -> float:
        """Adjust similarity threshold based on quality.
        
        Args:
            base_threshold: Base similarity threshold
            quality_factor: Overall quality factor
            candidate_quality: Candidate quality metrics
            
        Returns:
            Adjusted similarity threshold
        """
        # Lower threshold for high-quality candidates (more permissive)
        # Higher threshold for low-quality candidates (more restrictive)
        
        if quality_factor > 1.0:
            # High quality - be more permissive
            adjustment = -0.1 * (quality_factor - 1.0)
        else:
            # Low quality - be more restrictive
            adjustment = 0.15 * (1.0 - quality_factor)
        
        # Consider quality variance - high variance means inconsistent quality
        variance_penalty = min(0.05, candidate_quality['quality_variance'] * 0.1)
        
        adjusted_threshold = base_threshold + adjustment + variance_penalty
        
        # Ensure reasonable bounds
        return max(0.3, min(0.9, adjusted_threshold))
    
    def _adjust_path_score_threshold(self, 
                                   base_threshold: float,
                                   quality_factor: float,
                                   candidate_quality: Dict[str, float]) -> float:
        """Adjust path score threshold based on quality.
        
        Args:
            base_threshold: Base path score threshold
            quality_factor: Overall quality factor
            candidate_quality: Candidate quality metrics
            
        Returns:
            Adjusted path score threshold
        """
        # Similar logic to similarity threshold but more conservative
        if quality_factor > 1.0:
            adjustment = -0.05 * (quality_factor - 1.0)
        else:
            adjustment = 0.1 * (1.0 - quality_factor)
        
        # Consider relation density - more relations allow lower threshold
        if candidate_quality['relation_density'] > self.quality_stats['mean_relations']:
            adjustment -= 0.05
        
        adjusted_threshold = base_threshold + adjustment
        
        # Ensure reasonable bounds
        return max(0.2, min(0.8, adjusted_threshold))
    
    def _adjust_ranking_parameters(self, 
                                 quality_factor: float,
                                 candidate_quality: Dict[str, float]) -> Dict[str, float]:
        """Adjust ranking parameters based on quality.
        
        Args:
            quality_factor: Overall quality factor
            candidate_quality: Candidate quality metrics
            
        Returns:
            Dictionary of adjusted ranking parameters
        """
        # Base weights
        base_entity_weight = 0.3
        base_relation_weight = 0.25
        base_temporal_weight = 0.2
        base_coherence_weight = 0.25
        base_diversity_penalty = 0.1
        
        # Adjust based on candidate characteristics
        if candidate_quality['entity_density'] > self.quality_stats['mean_entities']:
            # High entity density - increase entity weight
            entity_weight = base_entity_weight * 1.2
        else:
            # Low entity density - decrease entity weight
            entity_weight = base_entity_weight * 0.8
        
        if candidate_quality['relation_density'] > self.quality_stats['mean_relations']:
            # High relation density - increase relation weight
            relation_weight = base_relation_weight * 1.3
        else:
            # Low relation density - decrease relation weight
            relation_weight = base_relation_weight * 0.7
        
        # Adjust temporal weight based on overall quality
        if quality_factor > 1.0:
            temporal_weight = base_temporal_weight * 1.1
        else:
            temporal_weight = base_temporal_weight * 0.9
        
        # Adjust coherence weight - more important for low quality
        if quality_factor < 1.0:
            coherence_weight = base_coherence_weight * 1.2
        else:
            coherence_weight = base_coherence_weight * 0.9
        
        # Adjust diversity penalty based on quality variance
        if candidate_quality['quality_variance'] > 0.1:
            diversity_penalty = base_diversity_penalty * 1.3
        else:
            diversity_penalty = base_diversity_penalty * 0.8
        
        # Normalize weights to sum to 1.0 (excluding diversity penalty)
        total_weight = entity_weight + relation_weight + temporal_weight + coherence_weight
        if total_weight > 0:
            entity_weight /= total_weight
            relation_weight /= total_weight
            temporal_weight /= total_weight
            coherence_weight /= total_weight
        
        return {
            'entity_weight': entity_weight,
            'relation_weight': relation_weight,
            'temporal_weight': temporal_weight,
            'coherence_weight': coherence_weight,
            'diversity_penalty': diversity_penalty
        }
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """Get insights about the quality distribution.
        
        Returns:
            Dictionary containing quality insights
        """
        return {
            'total_notes': len(self.atomic_notes),
            'quality_stats': self.quality_stats,
            'recommendations': self._generate_quality_recommendations()
        }
    
    def _generate_quality_recommendations(self) -> List[str]:
        """Generate recommendations based on quality analysis.
        
        Returns:
            List of quality improvement recommendations
        """
        recommendations = []
        
        if self.quality_stats['mean_quality'] < 0.5:
            recommendations.append("Overall note quality is low. Consider improving entity extraction and relation identification.")
        
        if self.quality_stats['mean_entities'] < 2:
            recommendations.append("Low entity density detected. Enhance entity recognition processes.")
        
        if self.quality_stats['mean_relations'] < 1:
            recommendations.append("Few relations detected. Improve relation extraction algorithms.")
        
        if self.quality_stats['std_quality'] > 0.3:
            recommendations.append("High quality variance detected. Standardize note processing pipeline.")
        
        if self.quality_stats['mean_content_length'] < 100:
            recommendations.append("Short content length detected. Consider including more context in notes.")
        
        return recommendations
    
    def adjust_retrieval_params(self, 
                              query_keywords: List[str] = None,
                              query_entities: List[str] = None,
                              current_top_k: int = 10) -> Dict[str, Any]:
        """Adjust retrieval parameters based on query characteristics.
        
        Args:
            query_keywords: Keywords from the query
            query_entities: Entities from the query
            current_top_k: Current top-k value
            
        Returns:
            Dictionary of adjusted retrieval parameters
        """
        # Analyze query complexity
        query_complexity = self._analyze_query_complexity(query_keywords, query_entities)
        
        # Adjust top_k based on complexity and quality
        adjusted_top_k = self._adjust_top_k(current_top_k, query_complexity)
        
        # Adjust path score threshold
        adjusted_path_score = self._adjust_path_score_for_retrieval(query_complexity)
        
        # Adjust similarity threshold
        adjusted_similarity = self._adjust_similarity_for_retrieval(query_complexity)
        
        return {
            'top_k': adjusted_top_k,
            'min_path_score': adjusted_path_score,
            'similarity_threshold': adjusted_similarity,
            'query_complexity': query_complexity
        }
    
    def _analyze_query_complexity(self, 
                                query_keywords: List[str] = None,
                                query_entities: List[str] = None) -> Dict[str, float]:
        """Analyze the complexity of the query.
        
        Args:
            query_keywords: Keywords from the query
            query_entities: Entities from the query
            
        Returns:
            Dictionary containing query complexity metrics
        """
        keywords = query_keywords or []
        entities = query_entities or []
        
        # Calculate complexity factors
        keyword_complexity = min(len(keywords) / 10.0, 1.0)  # Normalize to [0,1]
        entity_complexity = min(len(entities) / 5.0, 1.0)
        
        # Overall complexity score
        overall_complexity = (keyword_complexity + entity_complexity) / 2.0
        
        return {
            'keyword_complexity': keyword_complexity,
            'entity_complexity': entity_complexity,
            'overall_complexity': overall_complexity,
            'num_keywords': len(keywords),
            'num_entities': len(entities)
        }
    
    def _adjust_top_k(self, current_top_k: int, query_complexity: Dict[str, float]) -> int:
        """Adjust top_k based on query complexity and note quality.
        
        Args:
            current_top_k: Current top-k value
            query_complexity: Query complexity metrics
            
        Returns:
            Adjusted top-k value
        """
        base_adjustment = 1.0
        
        # Adjust based on query complexity
        if query_complexity['overall_complexity'] > 0.7:
            # High complexity - increase top_k
            base_adjustment = 1.3
        elif query_complexity['overall_complexity'] < 0.3:
            # Low complexity - decrease top_k
            base_adjustment = 0.8
        
        # Adjust based on note quality
        if self.quality_stats['mean_quality'] < 0.5:
            # Low quality notes - increase top_k to get more candidates
            base_adjustment *= 1.2
        elif self.quality_stats['mean_quality'] > 0.8:
            # High quality notes - can use fewer candidates
            base_adjustment *= 0.9
        
        adjusted_top_k = int(current_top_k * base_adjustment)
        
        # Ensure reasonable bounds
        return max(5, min(50, adjusted_top_k))
    
    def _adjust_path_score_for_retrieval(self, query_complexity: Dict[str, float]) -> float:
        """Adjust path score threshold for retrieval.
        
        Args:
            query_complexity: Query complexity metrics
            
        Returns:
            Adjusted path score threshold
        """
        base_threshold = 0.5
        
        # Adjust based on complexity
        if query_complexity['overall_complexity'] > 0.7:
            # Complex queries - lower threshold to be more inclusive
            adjustment = -0.1
        elif query_complexity['overall_complexity'] < 0.3:
            # Simple queries - higher threshold to be more selective
            adjustment = 0.1
        else:
            adjustment = 0.0
        
        # Adjust based on note quality
        if self.quality_stats['mean_quality'] < 0.5:
            # Low quality - be more selective
            adjustment += 0.05
        elif self.quality_stats['mean_quality'] > 0.8:
            # High quality - can be more inclusive
            adjustment -= 0.05
        
        adjusted_threshold = base_threshold + adjustment
        
        # Ensure reasonable bounds
        return max(0.2, min(0.8, adjusted_threshold))
    
    def _adjust_similarity_for_retrieval(self, query_complexity: Dict[str, float]) -> float:
        """Adjust similarity threshold for retrieval.
        
        Args:
            query_complexity: Query complexity metrics
            
        Returns:
            Adjusted similarity threshold
        """
        base_threshold = 0.7
        
        # Adjust based on complexity
        if query_complexity['overall_complexity'] > 0.7:
            # Complex queries - lower threshold
            adjustment = -0.1
        elif query_complexity['overall_complexity'] < 0.3:
            # Simple queries - higher threshold
            adjustment = 0.05
        else:
            adjustment = 0.0
        
        # Adjust based on note quality variance
        if self.quality_stats['std_quality'] > 0.3:
            # High variance - be more selective
            adjustment += 0.05
        
        adjusted_threshold = base_threshold + adjustment
        
        # Ensure reasonable bounds
        return max(0.4, min(0.9, adjusted_threshold))