"""Context packing module for preparing prompts with passage tracking."""

from typing import Any, Dict, List, Tuple, Optional
from loguru import logger

# Import new structure-based packer
from .structure_pack import create_structure_packer
from reasoning.qa_coverage import create_qa_coverage_scorer
from support.k_estimator import create_k_estimator


class ContextPacker:
    """Packs context notes into formatted prompts with passage tracking."""
    
    def __init__(self, calibration_path: Optional[str] = None):
        """Initialize context packer with new modules.
        
        Args:
            calibration_path: Path to calibration file for parameter loading
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
    
    def pack_context(self, notes: List[Dict[str, Any]], question: str, 
                    token_budget: int = 2000) -> Tuple[str, Dict[str, Dict], List[str]]:
        """Pack context notes into a formatted prompt with passage tracking.
        
        Args:
            notes: List of note dictionaries containing content and paragraph_idxs
            question: The question to ask
            token_budget: Maximum token budget for context
            
        Returns:
            Tuple of (packed_prompt, passages_by_idx, packed_order):
            - packed_prompt: The formatted prompt string
            - passages_by_idx: Dict mapping string idx to passage dict
            - packed_order: List of string indices in the order they appear in prompt
        """
        if self.use_legacy_packing:
            return self._pack_context_legacy(notes, question)
        
        try:
            # Convert notes to ranked paragraphs format
            ranked_paragraphs = self._convert_notes_to_paragraphs(notes)
            
            # Use structure packer for evidence selection and packing
            packed_text, passages_by_idx, packed_order = self.structure_packer.pack_evidence(
                question, ranked_paragraphs, token_budget
            )
            
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