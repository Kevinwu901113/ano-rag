"""Context packing module for preparing prompts with passage tracking."""

from typing import Any, Dict, List, Tuple, Optional, Set
from loguru import logger
import re
from collections import defaultdict

# Import new structure-based packer
from .structure_pack import create_structure_packer
from reasoning.qa_coverage import create_qa_coverage_scorer
from support.k_estimator import create_k_estimator


class ContextPacker:
    """Packs context notes into formatted prompts with passage tracking."""
    
    def __init__(self, calibration_path: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize context packer with new modules.
        
        Args:
            calibration_path: Path to calibration file for parameter loading
            config: Configuration dictionary for dual view packing
        """
        # Initialize new modules
        self.qa_coverage_scorer = create_qa_coverage_scorer(calibration_path=calibration_path)
        self.structure_packer = create_structure_packer(
            qa_coverage_scorer=self.qa_coverage_scorer,
            calibration_path=calibration_path
        )
        self.k_estimator = create_k_estimator(calibration_path=calibration_path)
        
        # Legacy mode flag for backward compatibility
        self.use_legacy_packing = False
        
        # Dual view packing configuration
        self.config = config or {}
        self.dual_view_config = self.config.get('context', {}).get('dual_view_packing', {})
    
    def pack_context(self, notes: List[Dict[str, Any]], question: str, 
                    token_budget: int = 2000, min_passages: int = 6) -> Tuple[str, Dict[str, Dict], List[str]]:
        """Pack context notes into a formatted prompt with passage tracking.
        
        Args:
            notes: List of note dictionaries containing content and paragraph_idxs
            question: The question to ask
            token_budget: Maximum token budget for context
            min_passages: Minimum number of passages to include (default: 6 for multi-hop)
            
        Returns:
            Tuple of (packed_prompt, passages_by_idx, packed_order):
            - packed_prompt: The formatted prompt string
            - passages_by_idx: Dict mapping string idx to passage dict
            - packed_order: List of string indices in the order they appear in prompt
        """
        # Check if dual view packing is enabled
        if self.dual_view_config.get('enabled', False):
            logger.info("Using dual view packing strategy")
            return self.pack_dual_view_context(notes, question, token_budget)
        
        if self.use_legacy_packing:
            return self._pack_context_legacy(notes, question)
        
        try:
            # Convert notes to ranked paragraphs format
            ranked_paragraphs = self._convert_notes_to_paragraphs(notes)
            
            # Use structure packer for evidence selection and packing
            packed_text, passages_by_idx, packed_order = self.structure_packer.pack_evidence(
                question, ranked_paragraphs, token_budget
            )
            
            # Ensure minimum passage count for multi-hop reasoning
            if len(packed_order) < min_passages and len(ranked_paragraphs) >= min_passages:
                logger.info(f"Expanding passages from {len(packed_order)} to {min_passages} for better multi-hop coverage")
                
                # Add more passages up to min_passages, prioritizing top-1 gets 2 sentences, others get 1
                additional_needed = min_passages - len(packed_order)
                available_paragraphs = [p for p in ranked_paragraphs if p['note_id'] not in packed_order]
                
                for i, paragraph in enumerate(available_paragraphs[:additional_needed]):
                    idx = paragraph['note_id']
                    content = paragraph['content']
                    
                    # Apply sentence selection: top-1 gets 2 sentences, others get 1
                    sentences = content.split('. ')
                    if i == 0 and len(packed_order) == 0:  # This is the new top-1
                        selected_content = '. '.join(sentences[:2]) if len(sentences) >= 2 else content
                    else:
                        selected_content = sentences[0] if sentences else content
                    
                    packed_order.append(idx)
                    passages_by_idx[idx] = {
                        'id': idx,
                        'content': selected_content,
                        'title': paragraph.get('title', ''),
                        'original_note': paragraph.get('original_note', paragraph)
                    }
                
                # Rebuild packed text with expanded passages
                context_parts = []
                for idx in packed_order:
                    passage = passages_by_idx[idx]
                    context_parts.append(f"[P{idx}] {passage['content']}")
                packed_text = "\n\n".join(context_parts)
            
            # Format the final prompt
            from llm.prompts import FINAL_ANSWER_PROMPT
            prompt = FINAL_ANSWER_PROMPT.format(context=packed_text, query=question)
            
            logger.info(f"Structured packing completed: {len(packed_order)} passages")
            return prompt, passages_by_idx, packed_order
            
        except Exception as e:
            logger.warning(f"Structured packing failed: {e}, falling back to legacy")
            return self._pack_context_legacy(notes, question)
    
    def _convert_notes_to_paragraphs(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert notes to ranked paragraphs format for structure packer.
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            List of paragraphs in structure packer format
        """
        paragraphs = []
        
        for i, note in enumerate(notes):
            # Extract relevant fields
            note_id = note.get('note_id', f'note_{i}')
            content = note.get('content', '')
            title = note.get('title', note.get('file_name', ''))
            
            # Get scores if available
            dense_score = note.get('similarity', note.get('final_score', 0.0))
            bm25_score = note.get('bm25_score', 0.0)
            
            paragraph = {
                'note_id': note_id,
                'id': note_id,
                'content': content,
                'title': title,
                'dense_score': dense_score,
                'bm25_score': bm25_score,
                'packed_position': i,
                'original_note': note
            }
            
            # Add embedding if available
            if 'embedding' in note:
                paragraph['embedding'] = note['embedding']
            
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _pack_context_legacy(self, notes: List[Dict[str, Any]], question: str) -> Tuple[str, Dict[str, Dict], List[str]]:
        """Legacy packing method for backward compatibility.
        
        Args:
            notes: List of note dictionaries
            question: The question to ask
            
        Returns:
            Tuple of (packed_prompt, passages_by_idx, packed_order)
        """
        context_parts: List[str] = []
        passages_by_idx: Dict[str, Dict] = {}
        packed_order: List[str] = []
        
        for i, note in enumerate(notes):
            # Extract paragraph_idxs from the note
            paragraph_idxs = note.get("paragraph_idxs", [])
            content = note.get("content", "")
            
            # If we have paragraph_idxs, use the first one as the primary idx
            if paragraph_idxs:
                primary_idx = str(paragraph_idxs[0])
            else:
                # Fallback: use note_id if no paragraph_idxs available
                note_id = note.get("note_id", "unknown")
                # Try to convert note_id to int, fallback to hash if not possible
                try:
                    idx = int(note_id) if isinstance(note_id, (int, str)) and str(note_id).isdigit() else hash(note_id) % 10000
                except:
                    idx = hash(str(note_id)) % 10000
                primary_idx = str(idx)
            
            context_parts.append(f"[P{primary_idx}] {content}")
            passages_by_idx[primary_idx] = {
                'id': primary_idx,
                'content': content,
                'title': note.get('title', ''),
                'original_note': note
            }
            packed_order.append(primary_idx)

        packed_text = "\n\n".join(context_parts)
        
        # Format the final prompt
        from llm.prompts import FINAL_ANSWER_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(context=packed_text, query=question)
        
        return prompt, passages_by_idx, packed_order
    
    def build_support_idxs(self, question: str, answer_text: str, 
                          passages_by_idx: Dict[str, Dict], 
                          packed_order: List[str]) -> List[int]:
        """Build support indices ensuring answer paragraph is first.
        
        Args:
            question: The input question
            answer_text: The generated answer text
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            List of support indices
        """
        try:
            # Use structure packer to build support indices
            support_idxs = self.structure_packer.build_support_idxs(
                question, answer_text, passages_by_idx, packed_order
            )
            return support_idxs
        except Exception as e:
            logger.warning(f"Failed to build support indices: {e}, using fallback")
            # Fallback: convert packed_order to integers
            fallback_idxs = []
            for idx_str in packed_order:
                try:
                    fallback_idxs.append(int(idx_str))
                except ValueError:
                    # If conversion fails, use hash
                    fallback_idxs.append(hash(idx_str) % 10000)
            return fallback_idxs
    
    def estimate_evidence_count(self, question: str, answer_idx: str, 
                              passages_by_idx: Dict[str, Dict], 
                              packed_order: List[str]) -> int:
        """Estimate the number of evidence paragraphs needed.
        
        Args:
            question: The input question
            answer_idx: Index of the answer paragraph
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            Estimated K value
        """
        try:
            k_estimate = self.k_estimator.estimate_K(
                question, answer_idx, passages_by_idx, packed_order
            )
            return k_estimate
        except Exception as e:
            logger.warning(f"Failed to estimate K: {e}, using default")
            return 2  # Default fallback
    
    def _extract_atomic_facts(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract atomic facts from notes for dual view packing.
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            List of atomic facts with metadata
        """
        facts = []
        
        for note in notes:
            # Extract atomic facts from note content
            content = note.get('content', '')
            note_id = note.get('note_id', '')
            
            # Check if note contains atomic facts structure
            if 'atomic_facts' in note:
                atomic_facts = note['atomic_facts']
                if isinstance(atomic_facts, list):
                    for i, fact in enumerate(atomic_facts):
                        if isinstance(fact, dict):
                            fact_data = {
                                'content': fact.get('fact', fact.get('content', '')),
                                'score': fact.get('importance', fact.get('score', 0.0)),
                                'entities': fact.get('entities', []),
                                'predicates': fact.get('predicates', []),
                                'temporal': fact.get('temporal', []),
                                'span': fact.get('span', {}),
                                'note_id': note_id,
                                'fact_id': f"{note_id}_fact_{i}",
                                'original_note': note
                            }
                            facts.append(fact_data)
            else:
                # Fallback: treat entire note content as a single fact
                fact_data = {
                    'content': content,
                    'score': note.get('similarity', note.get('final_score', 0.0)),
                    'entities': self._extract_entities_from_text(content),
                    'predicates': self._extract_predicates_from_text(content),
                    'temporal': self._extract_temporal_from_text(content),
                    'span': {'start': 0, 'end': len(content)},
                    'note_id': note_id,
                    'fact_id': f"{note_id}_fact_0",
                    'original_note': note
                }
                facts.append(fact_data)
        
        return facts
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction from text."""
        # Simple heuristic: capitalized words/phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _extract_predicates_from_text(self, text: str) -> List[str]:
        """Simple predicate extraction from text."""
        # Simple heuristic: common verbs and relations
        predicates = re.findall(r'\b(?:is|are|was|were|has|have|had|contains|includes|shows|indicates|suggests)\b', text.lower())
        return list(set(predicates))
    
    def _extract_temporal_from_text(self, text: str) -> List[str]:
        """Simple temporal extraction from text."""
        # Simple heuristic: years, dates, time expressions
        temporal = re.findall(r'\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b(?:today|yesterday|tomorrow|now|then|before|after)\b', text)
        return list(set(temporal))
    
    def _select_diverse_facts(self, facts: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """Select diverse facts based on score, coverage, and diversity.
        
        Args:
            facts: List of atomic facts
            question: The input question
            
        Returns:
            Selected facts for evidence list
        """
        fact_config = self.dual_view_config.get('fact_selection', {})
        min_score = fact_config.get('min_score', 0.5)
        min_coverage = fact_config.get('min_coverage', 0.3)
        max_facts = fact_config.get('max_facts', 20)
        diversity_weight = fact_config.get('diversity_weight', 0.4)
        
        # Filter facts by minimum score
        filtered_facts = [f for f in facts if f['score'] >= min_score]
        
        if not filtered_facts:
            # Fallback: take top facts by score
            filtered_facts = sorted(facts, key=lambda x: x['score'], reverse=True)[:max_facts]
        
        # Calculate diversity scores
        entity_counts = defaultdict(int)
        predicate_counts = defaultdict(int)
        temporal_counts = defaultdict(int)
        
        for fact in filtered_facts:
            for entity in fact['entities']:
                entity_counts[entity] += 1
            for predicate in fact['predicates']:
                predicate_counts[predicate] += 1
            for temporal in fact['temporal']:
                temporal_counts[temporal] += 1
        
        # Score facts with diversity consideration
        scored_facts = []
        for fact in filtered_facts:
            base_score = fact['score']
            
            # Calculate diversity bonus
            diversity_bonus = 0.0
            if fact_config.get('entity_diversity', True):
                entity_diversity = sum(1.0 / max(entity_counts[e], 1) for e in fact['entities'])
                diversity_bonus += entity_diversity
            
            if fact_config.get('predicate_diversity', True):
                predicate_diversity = sum(1.0 / max(predicate_counts[p], 1) for p in fact['predicates'])
                diversity_bonus += predicate_diversity
            
            if fact_config.get('temporal_diversity', True):
                temporal_diversity = sum(1.0 / max(temporal_counts[t], 1) for t in fact['temporal'])
                diversity_bonus += temporal_diversity
            
            final_score = base_score + diversity_weight * diversity_bonus
            scored_facts.append((final_score, fact))
        
        # Sort by final score and select top facts
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        selected_facts = [fact for _, fact in scored_facts[:max_facts]]
        
        logger.info(f"Selected {len(selected_facts)} diverse facts from {len(facts)} total facts")
        return selected_facts
    
    def _select_span_aligned_passages(self, notes: List[Dict[str, Any]], selected_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select original text spans aligned with selected facts.
        
        Args:
            notes: Original notes
            selected_facts: Selected atomic facts
            
        Returns:
            Selected text spans for verification
        """
        span_config = self.dual_view_config.get('span_alignment', {})
        max_spans = span_config.get('max_spans', 10)
        min_span_length = span_config.get('min_span_length', 20)
        max_span_length = span_config.get('max_span_length', 200)
        overlap_threshold = span_config.get('overlap_threshold', 0.8)
        
        spans = []
        used_note_ids = set()
        
        # Collect spans from selected facts
        for fact in selected_facts:
            note_id = fact['note_id']
            if note_id in used_note_ids:
                continue
            
            original_note = fact['original_note']
            content = original_note.get('content', '')
            
            # Extract span information
            span_info = fact.get('span', {})
            if span_info and 'start' in span_info and 'end' in span_info:
                start = span_info['start']
                end = span_info['end']
                span_text = content[start:end]
            else:
                # Fallback: use entire content or first part
                span_text = content[:max_span_length] if len(content) > max_span_length else content
            
            # Filter by length
            if len(span_text) >= min_span_length:
                span_data = {
                    'content': span_text,
                    'note_id': note_id,
                    'span_id': f"{note_id}_span",
                    'verification_score': fact['score'],
                    'original_note': original_note
                }
                spans.append(span_data)
                used_note_ids.add(note_id)
        
        # Sort by verification score and select top spans
        spans.sort(key=lambda x: x['verification_score'], reverse=True)
        selected_spans = spans[:max_spans]
        
        logger.info(f"Selected {len(selected_spans)} span-aligned passages from {len(spans)} candidates")
        return selected_spans
    
    def pack_dual_view_context(self, notes: List[Dict[str, Any]], question: str, 
                              token_budget: int = 2000) -> Tuple[str, Dict[str, Dict], List[str]]:
        """Pack context using dual view strategy: facts + original spans.
        
        Args:
            notes: List of note dictionaries
            question: The question to ask
            token_budget: Maximum token budget for context
            
        Returns:
            Tuple of (packed_prompt, passages_by_idx, packed_order)
        """
        if not self.dual_view_config.get('enabled', False):
            logger.info("Dual view packing disabled, falling back to default")
            return self.pack_context(notes, question, token_budget)
        
        facts_ratio = self.dual_view_config.get('facts_ratio', 0.7)
        original_ratio = self.dual_view_config.get('original_ratio', 0.3)
        
        facts_budget = int(token_budget * facts_ratio)
        original_budget = int(token_budget * original_ratio)
        
        logger.info(f"Dual view packing: facts_budget={facts_budget}, original_budget={original_budget}")
        
        # Extract and select atomic facts
        atomic_facts = self._extract_atomic_facts(notes)
        selected_facts = self._select_diverse_facts(atomic_facts, question)
        
        # Select span-aligned original passages
        selected_spans = self._select_span_aligned_passages(notes, selected_facts)
        
        # Build dual view context
        context_parts = []
        passages_by_idx = {}
        packed_order = []
        
        # Add facts section
        if selected_facts:
            context_parts.append("=== 证据事实清单 ===")
            for i, fact in enumerate(selected_facts):
                fact_id = fact['fact_id']
                fact_content = fact['content']
                context_parts.append(f"[F{i+1}] {fact_content}")
                passages_by_idx[fact_id] = {
                    'id': fact_id,
                    'content': fact_content,
                    'title': f"事实{i+1}",
                    'type': 'fact',
                    'original_note': fact['original_note']
                }
                packed_order.append(fact_id)
        
        # Add original spans section
        if selected_spans:
            context_parts.append("\n=== 原文片段 ===")
            for i, span in enumerate(selected_spans):
                span_id = span['span_id']
                span_content = span['content']
                context_parts.append(f"[S{i+1}] {span_content}")
                passages_by_idx[span_id] = {
                    'id': span_id,
                    'content': span_content,
                    'title': f"片段{i+1}",
                    'type': 'span',
                    'original_note': span['original_note']
                }
                packed_order.append(span_id)
        
        packed_text = "\n\n".join(context_parts)
        
        # Format the final prompt
        from llm.prompts import FINAL_ANSWER_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(context=packed_text, query=question)
        
        logger.info(f"Dual view packing completed: {len(selected_facts)} facts + {len(selected_spans)} spans")
        return prompt, passages_by_idx, packed_order