"""Context packer module for dual-view context assembly.

Supports both note view (atomic facts) and paragraph view (original text spans)
for optimized information density within token budget constraints.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from loguru import logger
import re
from collections import defaultdict
from dataclasses import dataclass

from config import config
from utils.text_utils import TextUtils


@dataclass
class FactNode:
    """Represents an atomic fact extracted from notes."""
    content: str
    entities: List[str]
    predicates: List[str]
    temporal_info: Optional[str]
    confidence_score: float
    source_note_id: str
    span_info: Optional[Dict[str, Any]] = None


@dataclass
class TextSpan:
    """Represents a text span from original document."""
    content: str
    start_pos: int
    end_pos: int
    source_note_id: str
    highlight_entities: List[str]
    relevance_score: float


@dataclass
class DualViewContext:
    """Container for dual-view context assembly result."""
    note_view: str  # Fact-based evidence list
    paragraph_view: str  # Original text spans
    total_tokens: int
    fact_count: int
    span_count: int
    coverage_entities: Set[str]


class ContextPacker:
    """Context packer supporting dual-view assembly.
    
    Assembles context from atomic notes into two complementary views:
    1. Note view: High-density fact list covering diverse entities/predicates/temporal info
    2. Paragraph view: Original text spans for verifiability
    
    Prioritizes fact density while maintaining token budget constraints.
    """
    
    def __init__(self):
        """Initialize context packer with configuration."""
        # Token budget configuration
        self.default_token_budget = config.get('context_packer.token_budget', 2000)
        self.fact_view_ratio = config.get('context_packer.fact_view_ratio', 0.7)  # 70% for facts
        self.paragraph_view_ratio = config.get('context_packer.paragraph_view_ratio', 0.3)  # 30% for spans
        
        # Fact selection parameters
        self.max_facts_per_entity = config.get('context_packer.max_facts_per_entity', 3)
        self.min_confidence_threshold = config.get('context_packer.min_confidence_threshold', 0.6)
        self.diversity_weight = config.get('context_packer.diversity_weight', 0.3)
        
        # Span selection parameters
        self.max_span_length = config.get('context_packer.max_span_length', 200)
        self.min_span_length = config.get('context_packer.min_span_length', 50)
        self.span_overlap_threshold = config.get('context_packer.span_overlap_threshold', 0.5)
        
        logger.info(f"ContextPacker initialized with fact_ratio={self.fact_view_ratio}, "
                   f"paragraph_ratio={self.paragraph_view_ratio}")
    
    def pack_dual_view_context(
        self,
        atomic_notes: List[Dict[str, Any]],
        query: str,
        token_budget: Optional[int] = None
    ) -> DualViewContext:
        """Pack atomic notes into dual-view context.
        
        Args:
            atomic_notes: List of atomic note dictionaries
            query: Query string for relevance scoring
            token_budget: Token budget constraint (uses default if None)
            
        Returns:
            DualViewContext with assembled note and paragraph views
        """
        if token_budget is None:
            token_budget = self.default_token_budget
            
        logger.info(f"Packing dual-view context for {len(atomic_notes)} notes "
                   f"with budget {token_budget} tokens")
        
        # Extract fact nodes and text spans from atomic notes
        fact_nodes = self._extract_fact_nodes(atomic_notes)
        text_spans = self._extract_text_spans(atomic_notes)
        
        # Calculate token budgets for each view
        fact_budget = int(token_budget * self.fact_view_ratio)
        span_budget = int(token_budget * self.paragraph_view_ratio)
        
        # Select and rank facts for note view
        selected_facts = self._select_facts_for_note_view(
            fact_nodes, query, fact_budget
        )
        
        # Select and rank spans for paragraph view
        selected_spans = self._select_spans_for_paragraph_view(
            text_spans, query, span_budget, selected_facts
        )
        
        # Assemble dual-view context
        note_view = self._assemble_note_view(selected_facts)
        paragraph_view = self._assemble_paragraph_view(selected_spans)
        
        # Calculate coverage statistics
        coverage_entities = set()
        for fact in selected_facts:
            coverage_entities.update(fact.entities)
        for span in selected_spans:
            coverage_entities.update(span.highlight_entities)
        
        # Estimate total tokens (rough approximation)
        total_tokens = len(note_view.split()) + len(paragraph_view.split())
        
        result = DualViewContext(
            note_view=note_view,
            paragraph_view=paragraph_view,
            total_tokens=total_tokens,
            fact_count=len(selected_facts),
            span_count=len(selected_spans),
            coverage_entities=coverage_entities
        )
        
        logger.info(f"Dual-view context assembled: {result.fact_count} facts, "
                   f"{result.span_count} spans, {len(result.coverage_entities)} entities, "
                   f"{result.total_tokens} tokens")
        
        return result
    
    def _extract_fact_nodes(self, atomic_notes: List[Dict[str, Any]]) -> List[FactNode]:
        """Extract fact nodes from atomic notes."""
        fact_nodes = []
        
        for note in atomic_notes:
            # Extract entities and relations from note
            entities = note.get('entities', [])
            relations = note.get('relations', [])
            content = note.get('content', '')
            note_id = note.get('id', str(len(fact_nodes)))
            
            # Extract predicates from relations
            predicates = []
            if isinstance(relations, list):
                for rel in relations:
                    if isinstance(rel, dict) and 'predicate' in rel:
                        predicates.append(rel['predicate'])
                    elif isinstance(rel, str):
                        predicates.append(rel)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_info(content)
            
            # Calculate confidence score based on note quality
            confidence_score = self._calculate_fact_confidence(note)
            
            # Create fact node
            fact_node = FactNode(
                content=content,
                entities=entities,
                predicates=predicates,
                temporal_info=temporal_info,
                confidence_score=confidence_score,
                source_note_id=note_id,
                span_info=note.get('span_info')
            )
            
            fact_nodes.append(fact_node)
        
        logger.debug(f"Extracted {len(fact_nodes)} fact nodes")
        return fact_nodes
    
    def _extract_text_spans(self, atomic_notes: List[Dict[str, Any]]) -> List[TextSpan]:
        """Extract text spans from atomic notes for paragraph view."""
        text_spans = []
        
        for note in atomic_notes:
            # Get original text and span information
            original_text = note.get('original_text', note.get('content', ''))
            span_info = note.get('span_info', {})
            entities = note.get('entities', [])
            note_id = note.get('id', str(len(text_spans)))
            
            # Create text spans from original text
            if original_text and len(original_text) >= self.min_span_length:
                # Split into sentences for span extraction
                sentences = TextUtils.split_sentences(original_text)
                
                for i, sentence in enumerate(sentences):
                    if len(sentence) >= self.min_span_length:
                        # Calculate relevance score
                        relevance_score = self._calculate_span_relevance(sentence, entities)
                        
                        span = TextSpan(
                            content=sentence,
                            start_pos=span_info.get('start', 0) + i * 100,  # Approximate
                            end_pos=span_info.get('end', len(sentence)) + i * 100,
                            source_note_id=note_id,
                            highlight_entities=entities,
                            relevance_score=relevance_score
                        )
                        
                        text_spans.append(span)
        
        logger.debug(f"Extracted {len(text_spans)} text spans")
        return text_spans
    
    def _select_facts_for_note_view(
        self,
        fact_nodes: List[FactNode],
        query: str,
        token_budget: int
    ) -> List[FactNode]:
        """Select high-quality facts for note view with diversity optimization."""
        # Filter facts by confidence threshold
        qualified_facts = [
            fact for fact in fact_nodes 
            if fact.confidence_score >= self.min_confidence_threshold
        ]
        
        if not qualified_facts:
            logger.warning("No facts meet confidence threshold, using all facts")
            qualified_facts = fact_nodes
        
        # Score facts based on query relevance and diversity
        scored_facts = []
        entity_coverage = defaultdict(int)
        predicate_coverage = defaultdict(int)
        
        for fact in qualified_facts:
            # Calculate query relevance score
            relevance_score = self._calculate_query_relevance(fact.content, query)
            
            # Calculate diversity bonus (lower coverage = higher bonus)
            diversity_bonus = 0
            for entity in fact.entities:
                diversity_bonus += 1.0 / (1 + entity_coverage[entity])
            for predicate in fact.predicates:
                diversity_bonus += 1.0 / (1 + predicate_coverage[predicate])
            
            # Temporal information bonus
            temporal_bonus = 0.1 if fact.temporal_info else 0
            
            # Combined score
            combined_score = (
                relevance_score * (1 - self.diversity_weight) +
                diversity_bonus * self.diversity_weight +
                temporal_bonus +
                fact.confidence_score * 0.1
            )
            
            scored_facts.append((fact, combined_score))
            
            # Update coverage counters
            for entity in fact.entities:
                entity_coverage[entity] += 1
            for predicate in fact.predicates:
                predicate_coverage[predicate] += 1
        
        # Sort by combined score
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        
        # Select facts within token budget
        selected_facts = []
        current_tokens = 0
        
        for fact, score in scored_facts:
            fact_tokens = len(fact.content.split())
            if current_tokens + fact_tokens <= token_budget:
                selected_facts.append(fact)
                current_tokens += fact_tokens
            else:
                break
        
        logger.debug(f"Selected {len(selected_facts)} facts from {len(qualified_facts)} candidates")
        return selected_facts
    
    def _select_spans_for_paragraph_view(
        self,
        text_spans: List[TextSpan],
        query: str,
        token_budget: int,
        selected_facts: List[FactNode]
    ) -> List[TextSpan]:
        """Select text spans for paragraph view, avoiding redundancy with facts."""
        # Get entities covered by selected facts
        fact_entities = set()
        for fact in selected_facts:
            fact_entities.update(fact.entities)
        
        # Score spans based on complementarity and relevance
        scored_spans = []
        
        for span in text_spans:
            # Calculate query relevance
            relevance_score = self._calculate_query_relevance(span.content, query)
            
            # Calculate complementarity (prefer spans with new entities)
            new_entities = set(span.highlight_entities) - fact_entities
            complementarity_score = len(new_entities) / max(1, len(span.highlight_entities))
            
            # Length penalty for very long spans
            length_penalty = max(0, 1 - (len(span.content) - self.max_span_length) / self.max_span_length)
            
            # Combined score
            combined_score = (
                relevance_score * 0.4 +
                complementarity_score * 0.4 +
                span.relevance_score * 0.1 +
                length_penalty * 0.1
            )
            
            scored_spans.append((span, combined_score))
        
        # Sort by combined score
        scored_spans.sort(key=lambda x: x[1], reverse=True)
        
        # Select spans within token budget, avoiding overlap
        selected_spans = []
        current_tokens = 0
        used_content = set()
        
        for span, score in scored_spans:
            span_tokens = len(span.content.split())
            
            # Check for content overlap
            if self._has_significant_overlap(span.content, used_content):
                continue
            
            if current_tokens + span_tokens <= token_budget:
                selected_spans.append(span)
                current_tokens += span_tokens
                used_content.add(span.content)
            else:
                break
        
        logger.debug(f"Selected {len(selected_spans)} spans from {len(text_spans)} candidates")
        return selected_spans
    
    def _assemble_note_view(self, selected_facts: List[FactNode]) -> str:
        """Assemble note view from selected facts."""
        if not selected_facts:
            return "No factual evidence available."
        
        note_parts = []
        note_parts.append("## 事实证据清单\n")
        
        for i, fact in enumerate(selected_facts, 1):
            # Format fact with entities and temporal info
            fact_text = f"{i}. {fact.content}"
            
            # Add entity information
            if fact.entities:
                entities_str = ", ".join(fact.entities[:3])  # Limit to top 3
                fact_text += f" [实体: {entities_str}]"
            
            # Add temporal information
            if fact.temporal_info:
                fact_text += f" [时间: {fact.temporal_info}]"
            
            # Add confidence indicator
            if fact.confidence_score >= 0.8:
                fact_text += " ✓"
            
            note_parts.append(fact_text)
        
        return "\n".join(note_parts)
    
    def _assemble_paragraph_view(self, selected_spans: List[TextSpan]) -> str:
        """Assemble paragraph view from selected spans."""
        if not selected_spans:
            return "No original text spans available."
        
        paragraph_parts = []
        paragraph_parts.append("\n## 原文片段\n")
        
        for i, span in enumerate(selected_spans, 1):
            # Highlight entities in span content
            highlighted_content = self._highlight_entities_in_text(
                span.content, span.highlight_entities
            )
            
            span_text = f"[P{i}] {highlighted_content}"
            paragraph_parts.append(span_text)
        
        return "\n\n".join(paragraph_parts)
    
    def _extract_temporal_info(self, content: str) -> Optional[str]:
        """Extract temporal information from content."""
        # Simple temporal pattern matching
        temporal_patterns = [
            r'\d{4}年\d{1,2}月',  # 2023年12月
            r'\d{4}-\d{2}-\d{2}',  # 2023-12-01
            r'\d{1,2}月\d{1,2}日',  # 12月1日
            r'[去今明]年',  # 去年、今年、明年
            r'[上下]个?[月周年]',  # 上个月、下周、明年
        ]
        
        for pattern in temporal_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group()
        
        return None
    
    def _calculate_fact_confidence(self, note: Dict[str, Any]) -> float:
        """Calculate confidence score for a fact based on note quality."""
        confidence = 0.5  # Base confidence
        
        # Entity count bonus
        entities = note.get('entities', [])
        if len(entities) >= 2:
            confidence += 0.2
        elif len(entities) >= 1:
            confidence += 0.1
        
        # Relation count bonus
        relations = note.get('relations', [])
        if len(relations) >= 1:
            confidence += 0.1
        
        # Content length bonus (not too short, not too long)
        content = note.get('content', '')
        content_length = len(content)
        if 50 <= content_length <= 200:
            confidence += 0.1
        elif content_length > 200:
            confidence -= 0.1
        
        # Span info bonus (has source location)
        if note.get('span_info'):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_span_relevance(self, content: str, entities: List[str]) -> float:
        """Calculate relevance score for a text span."""
        if not entities:
            return 0.3  # Base relevance
        
        # Count entity mentions in content
        entity_mentions = 0
        content_lower = content.lower()
        
        for entity in entities:
            if entity.lower() in content_lower:
                entity_mentions += 1
        
        # Calculate relevance based on entity coverage
        relevance = 0.3 + (entity_mentions / len(entities)) * 0.7
        return min(1.0, relevance)
    
    def _calculate_query_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query."""
        if not query:
            return 0.5
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _has_significant_overlap(self, content: str, used_content: Set[str]) -> bool:
        """Check if content has significant overlap with already used content."""
        content_words = set(content.lower().split())
        
        for used in used_content:
            used_words = set(used.lower().split())
            
            # Calculate overlap ratio
            intersection = len(content_words & used_words)
            min_length = min(len(content_words), len(used_words))
            
            if min_length > 0 and intersection / min_length > self.span_overlap_threshold:
                return True
        
        return False
    
    def _highlight_entities_in_text(self, content: str, entities: List[str]) -> str:
        """Highlight entities in text content."""
        highlighted = content
        
        for entity in entities:
            # Simple highlighting with **bold** markers
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
            highlighted = pattern.sub(f"**{entity}**", highlighted)
        
        return highlighted
    
    def format_context_for_llm(
        self,
        atomic_notes: List[Dict[str, Any]],
        query: str,
        token_budget: Optional[int] = None,
        include_metadata: bool = True
    ) -> str:
        """Format dual-view context for LLM consumption.
        
        Args:
            atomic_notes: List of atomic note dictionaries
            query: Query string for relevance scoring
            token_budget: Token budget constraint
            include_metadata: Whether to include metadata in output
            
        Returns:
            Formatted context string ready for LLM
        """
        dual_context = self.pack_dual_view_context(atomic_notes, query, token_budget)
        
        # Combine note view and paragraph view
        formatted_parts = []
        
        # Add note view (facts)
        if dual_context.note_view:
            formatted_parts.append(dual_context.note_view)
        
        # Add paragraph view (original text)
        if dual_context.paragraph_view:
            formatted_parts.append(dual_context.paragraph_view)
        
        # Add metadata if requested
        if include_metadata:
            metadata_info = (
                f"\n## 上下文统计\n"
                f"- 事实数量: {dual_context.fact_count}\n"
                f"- 原文片段: {dual_context.span_count}\n"
                f"- 覆盖实体: {len(dual_context.coverage_entities)}\n"
                f"- 预估Token: {dual_context.total_tokens}"
            )
            formatted_parts.append(metadata_info)
        
        return "\n\n".join(formatted_parts)
    
    def get_context_summary(self, dual_context: DualViewContext) -> Dict[str, Any]:
        """Get summary statistics for dual-view context.
        
        Args:
            dual_context: DualViewContext result
            
        Returns:
            Dictionary with context statistics
        """
        return {
            'fact_count': dual_context.fact_count,
            'span_count': dual_context.span_count,
            'total_tokens': dual_context.total_tokens,
            'coverage_entities': list(dual_context.coverage_entities),
            'entity_count': len(dual_context.coverage_entities),
            'fact_density': dual_context.fact_count / max(1, dual_context.total_tokens),
            'has_facts': dual_context.fact_count > 0,
            'has_spans': dual_context.span_count > 0
        }
    
    def optimize_token_allocation(
        self,
        atomic_notes: List[Dict[str, Any]],
        query: str,
        target_token_budget: int,
        iterations: int = 3
    ) -> Tuple[DualViewContext, Dict[str, Any]]:
        """Optimize token allocation between fact and paragraph views.
        
        Args:
            atomic_notes: List of atomic note dictionaries
            query: Query string for relevance scoring
            target_token_budget: Target token budget
            iterations: Number of optimization iterations
            
        Returns:
            Tuple of (optimized_context, optimization_stats)
        """
        best_context = None
        best_score = -1
        optimization_stats = {
            'iterations': [],
            'best_allocation': None,
            'convergence': False
        }
        
        # Try different fact/paragraph ratios
        ratios_to_try = [
            (0.8, 0.2),  # Heavy on facts
            (0.7, 0.3),  # Default
            (0.6, 0.4),  # Balanced
            (0.5, 0.5),  # Equal
            (0.4, 0.6),  # Heavy on paragraphs
        ]
        
        for iteration in range(iterations):
            for fact_ratio, para_ratio in ratios_to_try:
                # Temporarily adjust ratios
                original_fact_ratio = self.fact_view_ratio
                original_para_ratio = self.paragraph_view_ratio
                
                self.fact_view_ratio = fact_ratio
                self.paragraph_view_ratio = para_ratio
                
                try:
                    # Generate context with current ratios
                    context = self.pack_dual_view_context(
                        atomic_notes, query, target_token_budget
                    )
                    
                    # Calculate optimization score
                    score = self._calculate_optimization_score(context, query)
                    
                    # Track iteration stats
                    iteration_stats = {
                        'fact_ratio': fact_ratio,
                        'para_ratio': para_ratio,
                        'score': score,
                        'fact_count': context.fact_count,
                        'span_count': context.span_count,
                        'total_tokens': context.total_tokens
                    }
                    optimization_stats['iterations'].append(iteration_stats)
                    
                    # Update best if this is better
                    if score > best_score:
                        best_score = score
                        best_context = context
                        optimization_stats['best_allocation'] = {
                            'fact_ratio': fact_ratio,
                            'para_ratio': para_ratio,
                            'score': score
                        }
                
                finally:
                    # Restore original ratios
                    self.fact_view_ratio = original_fact_ratio
                    self.paragraph_view_ratio = original_para_ratio
        
        # Check for convergence
        if len(optimization_stats['iterations']) >= 2:
            recent_scores = [it['score'] for it in optimization_stats['iterations'][-3:]]
            if len(set(recent_scores)) == 1:  # All recent scores are the same
                optimization_stats['convergence'] = True
        
        logger.info(f"Token allocation optimization completed. "
                   f"Best score: {best_score:.3f}, "
                   f"Best allocation: {optimization_stats['best_allocation']}")
        
        return best_context or self.pack_dual_view_context(atomic_notes, query, target_token_budget), optimization_stats
    
    def _calculate_optimization_score(self, context: DualViewContext, query: str) -> float:
        """Calculate optimization score for a context configuration.
        
        Higher scores indicate better context quality.
        """
        if context.total_tokens == 0:
            return 0.0
        
        # Fact density score (more facts per token is better)
        fact_density = context.fact_count / context.total_tokens
        
        # Entity coverage score (more unique entities is better)
        entity_coverage = len(context.coverage_entities)
        
        # Balance score (having both facts and spans is better than just one)
        balance_score = 1.0
        if context.fact_count == 0 or context.span_count == 0:
            balance_score = 0.5
        
        # Query relevance score (rough approximation)
        query_words = set(query.lower().split())
        note_words = set(context.note_view.lower().split())
        para_words = set(context.paragraph_view.lower().split())
        all_words = note_words | para_words
        
        query_relevance = 0.5
        if query_words and all_words:
            query_relevance = len(query_words & all_words) / len(query_words | all_words)
        
        # Combined optimization score
        optimization_score = (
            fact_density * 0.3 +
            entity_coverage * 0.01 +  # Normalized by typical entity count
            balance_score * 0.2 +
            query_relevance * 0.5
        )
        
        return optimization_score