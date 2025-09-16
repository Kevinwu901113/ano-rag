"""Structure-based evidence packing module.

This module implements pure structural constraints and data-driven sorting for evidence packing.
It ensures "answer paragraph + bridge paragraph" structure using QA coverage scores and similarity ranking.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
from loguru import logger
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class StructurePacker:
    """Structure-based evidence packing system."""
    
    def __init__(self, qa_coverage_scorer=None, calibration_path: Optional[str] = None):
        """
        Initialize the structure packer.
        
        Args:
            qa_coverage_scorer: QA coverage scorer instance
            calibration_path: Path to calibration file
        """
        self.qa_coverage_scorer = qa_coverage_scorer
        self.calibration_path = calibration_path
        
        # Default parameters (can be overridden by calibration)
        self.max_sentences_per_paragraph = 2
        self.similarity_threshold = 0.3
        self.bridge_selection_method = "graph"  # "graph" or "similarity"
        self.mmr_lambda = 0.7  # For MMR sentence selection
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def pack_evidence(self, question: str, ranked_paragraphs: List[Dict[str, Any]], 
                     token_budget: int = 2000) -> Tuple[str, Dict[str, Dict], List[str]]:
        """
        Pack evidence using structural constraints and data-driven sorting.
        
        Args:
            question: The input question
            ranked_paragraphs: List of ranked paragraphs with scores
            token_budget: Maximum token budget for context
            
        Returns:
            Tuple of (packed_text, passages_by_idx, packed_order)
        """
        # P2-7: 候选为空时强制保留top-1兜底
        if not ranked_paragraphs:
            logger.warning("No ranked paragraphs provided, returning empty result")
            return "", {}, []
        
        logger.info(f"Packing evidence for question with {len(ranked_paragraphs)} candidates")
        
        # Step 1: Determine answer paragraph
        answer_paragraph = self._select_answer_paragraph(question, ranked_paragraphs)
        
        # Step 2: Build paragraph graph
        paragraph_graph = self._build_paragraph_graph(ranked_paragraphs)
        
        # Step 3: Select bridge paragraphs
        bridge_paragraphs = self._select_bridge_paragraphs(
            question, answer_paragraph, ranked_paragraphs, paragraph_graph
        )
        
        # Step 4: Combine and order paragraphs
        selected_paragraphs = [answer_paragraph] + bridge_paragraphs
        
        # Step 5: Select sentences within each paragraph
        packed_content = []
        passages_by_idx = {}
        packed_order = []
        current_tokens = 0
        
        for i, paragraph in enumerate(selected_paragraphs):
            if current_tokens >= token_budget:
                break
            
            # Select best sentences from paragraph
            selected_sentences = self._select_sentences_from_paragraph(
                question, paragraph, max_sentences=self.max_sentences_per_paragraph
            )
            
            if selected_sentences:
                paragraph_text = " ".join(selected_sentences)
                paragraph_tokens = len(paragraph_text.split())
                
                if current_tokens + paragraph_tokens <= token_budget:
                    paragraph_id = paragraph.get('note_id', paragraph.get('id', f'p_{i}'))
                    
                    # Format as [P{idx}] content
                    formatted_text = f"[P{i}] {paragraph_text}"
                    packed_content.append(formatted_text)
                    
                    # Store in passages_by_idx
                    passages_by_idx[str(i)] = {
                        'id': paragraph_id,
                        'content': paragraph_text,
                        'title': paragraph.get('title', ''),
                        'original_paragraph': paragraph
                    }
                    
                    packed_order.append(str(i))
                    current_tokens += paragraph_tokens
        
        # P2-7: 如果被过滤到0段落，强制保留top-1作为兜底
        if not packed_content and ranked_paragraphs:
            logger.warning("All paragraphs were filtered out, forcing top-1 as fallback")
            top_paragraph = ranked_paragraphs[0]
            
            # 选择top-1段落的最佳句子
            selected_sentences = self._select_sentences_from_paragraph(
                question, top_paragraph, max_sentences=self.max_sentences_per_paragraph
            )
            
            if selected_sentences:
                paragraph_text = " ".join(selected_sentences)
            else:
                # 如果句子选择也失败，使用原始内容的前部分
                original_content = top_paragraph.get('content', top_paragraph.get('text', ''))
                words = original_content.split()
                paragraph_text = " ".join(words[:min(len(words), token_budget // 2)])
            
            paragraph_id = top_paragraph.get('note_id', top_paragraph.get('id', 'p_0'))
            formatted_text = f"[P0] {paragraph_text}"
            
            packed_content = [formatted_text]
            passages_by_idx = {
                '0': {
                    'id': paragraph_id,
                    'content': paragraph_text,
                    'title': top_paragraph.get('title', ''),
                    'original_paragraph': top_paragraph
                }
            }
            packed_order = ['0']
            current_tokens = len(paragraph_text.split())
        
        packed_text = "\n\n".join(packed_content)
        
        logger.info(f"Packed {len(packed_order)} paragraphs, {current_tokens} tokens")
        
        return packed_text, passages_by_idx, packed_order
    
    def _select_answer_paragraph(self, question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the answer paragraph from candidates.
        
        Args:
            question: The input question
            candidates: List of candidate paragraphs
            
        Returns:
            Selected answer paragraph
        """
        if not candidates:
            return {}
        
        # Use QA coverage scorer if available
        if self.qa_coverage_scorer:
            best_id, best_score = self.qa_coverage_scorer.best_answering_paragraph(question, candidates)
            
            # Find the paragraph with best_id
            for candidate in candidates:
                candidate_id = candidate.get('note_id', candidate.get('id', ''))
                if candidate_id == best_id:
                    logger.info(f"Selected answer paragraph {best_id} with QA score {best_score:.3f}")
                    return candidate
        
        # Fallback: use the highest ranked paragraph
        best_paragraph = candidates[0]
        logger.info(f"Selected answer paragraph (fallback): {best_paragraph.get('note_id', 'unknown')}")
        return best_paragraph
    
    def _build_paragraph_graph(self, paragraphs: List[Dict[str, Any]]) -> nx.Graph:
        """
        Build a graph of paragraphs based on similarity and entity overlap.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            NetworkX graph with paragraphs as nodes
        """
        graph = nx.Graph()
        
        # Add nodes
        for i, paragraph in enumerate(paragraphs):
            paragraph_id = paragraph.get('note_id', paragraph.get('id', f'p_{i}'))
            graph.add_node(paragraph_id, paragraph=paragraph, index=i)
        
        # Add edges based on similarity
        for i, para1 in enumerate(paragraphs):
            for j, para2 in enumerate(paragraphs[i+1:], i+1):
                similarity = self._compute_paragraph_similarity(para1, para2)
                
                if similarity > self.similarity_threshold:
                    id1 = para1.get('note_id', para1.get('id', f'p_{i}'))
                    id2 = para2.get('note_id', para2.get('id', f'p_{j}'))
                    graph.add_edge(id1, id2, weight=similarity)
        
        logger.info(f"Built paragraph graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _compute_paragraph_similarity(self, para1: Dict[str, Any], para2: Dict[str, Any]) -> float:
        """
        Compute similarity between two paragraphs.
        
        Args:
            para1: First paragraph
            para2: Second paragraph
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use embeddings if available
        if 'embedding' in para1 and 'embedding' in para2:
            emb1 = np.array(para1['embedding']).reshape(1, -1)
            emb2 = np.array(para2['embedding']).reshape(1, -1)
            return cosine_similarity(emb1, emb2)[0, 0]
        
        # Fallback to text similarity
        content1 = para1.get('content', '')
        content2 = para2.get('content', '')
        
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
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _select_bridge_paragraphs(self, question: str, answer_paragraph: Dict[str, Any], 
                                 all_paragraphs: List[Dict[str, Any]], 
                                 graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        Select bridge paragraphs using graph structure or similarity.
        
        Args:
            question: The input question
            answer_paragraph: The selected answer paragraph
            all_paragraphs: All available paragraphs
            graph: Paragraph similarity graph
            
        Returns:
            List of selected bridge paragraphs
        """
        answer_id = answer_paragraph.get('note_id', answer_paragraph.get('id', ''))
        
        if self.bridge_selection_method == "graph" and answer_id in graph:
            return self._select_bridge_by_graph(answer_id, graph, max_bridges=2)
        else:
            return self._select_bridge_by_similarity(question, answer_paragraph, all_paragraphs, max_bridges=2)
    
    def _select_bridge_by_graph(self, answer_id: str, graph: nx.Graph, max_bridges: int = 2) -> List[Dict[str, Any]]:
        """
        Select bridge paragraphs using graph shortest paths.
        
        Args:
            answer_id: ID of the answer paragraph
            graph: Paragraph similarity graph
            max_bridges: Maximum number of bridge paragraphs
            
        Returns:
            List of bridge paragraphs
        """
        bridge_paragraphs = []
        
        if answer_id not in graph:
            return bridge_paragraphs
        
        # Get neighbors sorted by edge weight (similarity)
        neighbors = []
        for neighbor in graph.neighbors(answer_id):
            edge_data = graph.get_edge_data(answer_id, neighbor)
            weight = edge_data.get('weight', 0.0)
            paragraph = graph.nodes[neighbor]['paragraph']
            neighbors.append((weight, paragraph))
        
        # Sort by weight (descending) and select top bridges
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        for i, (weight, paragraph) in enumerate(neighbors[:max_bridges]):
            bridge_paragraphs.append(paragraph)
            logger.info(f"Selected bridge paragraph {i+1} with similarity {weight:.3f}")
        
        return bridge_paragraphs
    
    def _select_bridge_by_similarity(self, question: str, answer_paragraph: Dict[str, Any], 
                                   all_paragraphs: List[Dict[str, Any]], 
                                   max_bridges: int = 2) -> List[Dict[str, Any]]:
        """
        Select bridge paragraphs by question similarity.
        
        Args:
            question: The input question
            answer_paragraph: The answer paragraph
            all_paragraphs: All available paragraphs
            max_bridges: Maximum number of bridge paragraphs
            
        Returns:
            List of bridge paragraphs
        """
        answer_id = answer_paragraph.get('note_id', answer_paragraph.get('id', ''))
        
        # Score all paragraphs by question similarity
        scored_paragraphs = []
        for paragraph in all_paragraphs:
            paragraph_id = paragraph.get('note_id', paragraph.get('id', ''))
            
            # Skip the answer paragraph
            if paragraph_id == answer_id:
                continue
            
            content = paragraph.get('content', '')
            similarity = self._compute_text_similarity(question, content)
            scored_paragraphs.append((similarity, paragraph))
        
        # Sort by similarity and select top bridges
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        
        bridge_paragraphs = []
        for i, (similarity, paragraph) in enumerate(scored_paragraphs[:max_bridges]):
            bridge_paragraphs.append(paragraph)
            logger.info(f"Selected bridge paragraph {i+1} with question similarity {similarity:.3f}")
        
        return bridge_paragraphs
    
    def _select_sentences_from_paragraph(self, question: str, paragraph: Dict[str, Any], 
                                       max_sentences: int = 2) -> List[str]:
        """
        Select the best sentences from a paragraph using MMR.
        
        Args:
            question: The input question
            paragraph: The paragraph to select from
            max_sentences: Maximum number of sentences to select
            
        Returns:
            List of selected sentences
        """
        content = paragraph.get('content', '')
        if not content:
            return []
        
        # Split into sentences
        sentences = self._split_sentences(content)
        if not sentences:
            return []
        
        if len(sentences) <= max_sentences:
            return sentences
        
        # Score sentences by QA coverage
        sentence_scores = []
        for sentence in sentences:
            if self.qa_coverage_scorer:
                score = self.qa_coverage_scorer.score_sentence(question, sentence)
            else:
                score = self._compute_text_similarity(question, sentence)
            sentence_scores.append(score)
        
        # Apply MMR for diversity
        selected_sentences = self._mmr_sentence_selection(
            sentences, sentence_scores, max_sentences
        )
        
        return selected_sentences
    
    def _mmr_sentence_selection(self, sentences: List[str], scores: List[float], 
                              max_sentences: int) -> List[str]:
        """
        Select sentences using Maximal Marginal Relevance (MMR).
        
        Args:
            sentences: List of candidate sentences
            scores: Relevance scores for each sentence
            max_sentences: Maximum number of sentences to select
            
        Returns:
            List of selected sentences
        """
        if not sentences or max_sentences <= 0:
            return []
        
        selected = []
        remaining_indices = list(range(len(sentences)))
        
        # Select first sentence with highest score
        best_idx = max(remaining_indices, key=lambda i: scores[i])
        selected.append(sentences[best_idx])
        remaining_indices.remove(best_idx)
        
        # Select remaining sentences using MMR
        while len(selected) < max_sentences and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = scores[idx]
                
                # Compute max similarity to already selected sentences
                max_similarity = 0.0
                for selected_sentence in selected:
                    similarity = self._compute_text_similarity(sentences[idx], selected_sentence)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: λ * relevance - (1-λ) * max_similarity
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity
                mmr_scores.append((mmr_score, idx))
            
            # Select sentence with highest MMR score
            best_mmr_score, best_idx = max(mmr_scores, key=lambda x: x[0])
            selected.append(sentences[best_idx])
            remaining_indices.remove(best_idx)
        
        return selected
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences
    
    def build_support_idxs(self, question: str, answer_text: str, 
                          passages_by_idx: Dict[str, Dict], 
                          packed_order: List[str]) -> List[int]:
        """
        Build support indices ensuring answer paragraph is first.
        
        Args:
            question: The input question
            answer_text: The generated answer text
            passages_by_idx: Dictionary of passages by index
            packed_order: Order of packed passages
            
        Returns:
            List of support indices
        """
        if not packed_order:
            return []
        
        # Filter out ghost indices (only keep valid indices from passages_by_idx)
        valid_indices = [idx for idx in packed_order if idx in passages_by_idx.keys()]
        
        if not valid_indices:
            return []
        
        # Find answer paragraph (paragraph most likely to contain the answer)
        answer_paragraph_idx = self._find_answer_paragraph_idx(
            answer_text, passages_by_idx, valid_indices
        )
        
        # Build support list with answer paragraph first
        support_idxs = []
        
        if answer_paragraph_idx is not None:
            support_idxs.append(int(answer_paragraph_idx))
        
        # Add remaining paragraphs in order of proximity to answer paragraph
        remaining_indices = [idx for idx in valid_indices if idx != answer_paragraph_idx]
        
        # Sort remaining by proximity to answer paragraph or by original order
        if answer_paragraph_idx is not None:
            remaining_indices.sort(key=lambda idx: abs(int(idx) - int(answer_paragraph_idx)))
        
        for idx in remaining_indices:
            support_idxs.append(int(idx))
        
        logger.info(f"Built support indices: {support_idxs}")
        return support_idxs
    
    def _find_answer_paragraph_idx(self, answer_text: str, passages_by_idx: Dict[str, Dict], 
                                  valid_indices: List[str]) -> Optional[str]:
        """
        Find the paragraph index most likely to contain the answer.
        
        Args:
            answer_text: The generated answer
            passages_by_idx: Dictionary of passages
            valid_indices: List of valid indices
            
        Returns:
            Index of the answer paragraph or None
        """
        if not answer_text or not valid_indices:
            return None
        
        best_idx = None
        best_score = 0.0
        
        for idx in valid_indices:
            passage = passages_by_idx.get(idx, {})
            content = passage.get('content', '')
            
            # Check if answer text appears in content
            if answer_text.lower() in content.lower():
                return idx
            
            # Otherwise, use similarity score
            similarity = self._compute_text_similarity(answer_text, content)
            if similarity > best_score:
                best_score = similarity
                best_idx = idx
        
        return best_idx
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load structure packing parameters
            structure_config = calibration.get('structure_packing', {})
            self.max_sentences_per_paragraph = structure_config.get('max_sentences_per_paragraph', 2)
            self.similarity_threshold = structure_config.get('similarity_threshold', 0.3)
            self.bridge_selection_method = structure_config.get('bridge_selection_method', 'graph')
            self.mmr_lambda = structure_config.get('mmr_lambda', 0.7)
            
            logger.info(f"Structure packing calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load structure packing calibration: {e}")


def create_structure_packer(qa_coverage_scorer=None, calibration_path: Optional[str] = None) -> StructurePacker:
    """
    Create a StructurePacker instance.
    
    Args:
        qa_coverage_scorer: QA coverage scorer instance
        calibration_path: Path to calibration file
        
    Returns:
        StructurePacker instance
    """
    return StructurePacker(qa_coverage_scorer=qa_coverage_scorer, calibration_path=calibration_path)