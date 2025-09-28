"""Span picker module for learnable answer localization.

This module implements learnable span selection using cross-encoder scoring
to replace rule-based answer extraction. It generates span candidates using
general heuristics and scores them with a trained model.
"""

import json
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle


class SpanPicker:
    """Learnable span selection system for answer extraction."""
    
    def __init__(self, model_type: str = "cross_encoder", calibration_path: Optional[str] = None):
        """
        Initialize the span picker.
        
        Args:
            model_type: Type of model to use ('cross_encoder' or 'simple')
            calibration_path: Path to calibration file
        """
        self.model_type = model_type
        self.calibration_path = calibration_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_span_length = 1
        self.max_span_length = 10
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def _generate_span_candidates(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Generate span candidates using general heuristics.
        
        Args:
            text: Input text to extract spans from
            
        Returns:
            List of (span_text, start_pos, end_pos) tuples
        """
        candidates = []
        words = text.split()
        
        if not words:
            return candidates
        
        # Generate n-gram spans (1 to max_span_length)
        for length in range(self.min_span_length, min(self.max_span_length + 1, len(words) + 1)):
            for start in range(len(words) - length + 1):
                span_words = words[start:start + length]
                span_text = " ".join(span_words)
                
                # Calculate character positions
                start_pos = len(" ".join(words[:start]))
                if start > 0:
                    start_pos += 1  # Account for space
                end_pos = start_pos + len(span_text)
                
                candidates.append((span_text, start_pos, end_pos))
        
        # Add quoted segments
        quoted_spans = self._extract_quoted_spans(text)
        candidates.extend(quoted_spans)
        
        # Add number and date patterns
        pattern_spans = self._extract_pattern_spans(text)
        candidates.extend(pattern_spans)
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for span_text, start_pos, end_pos in candidates:
            if span_text not in seen:
                seen.add(span_text)
                unique_candidates.append((span_text, start_pos, end_pos))
        
        return unique_candidates
    
    def _extract_quoted_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract spans within quotes.
        
        Args:
            text: Input text
            
        Returns:
            List of quoted spans
        """
        spans = []
        
        # Find content within double quotes
        for match in re.finditer(r'"([^"]+)"', text):
            span_text = match.group(1)
            start_pos = match.start(1)
            end_pos = match.end(1)
            spans.append((span_text, start_pos, end_pos))
        
        # Find content within single quotes
        for match in re.finditer(r"'([^']+)'", text):
            span_text = match.group(1)
            start_pos = match.start(1)
            end_pos = match.end(1)
            spans.append((span_text, start_pos, end_pos))
        
        return spans
    
    def _extract_pattern_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract spans matching common patterns (numbers, dates, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of pattern-based spans
        """
        spans = []
        
        # Numbers (including decimals)
        for match in re.finditer(r'\b\d+(?:\.\d+)?\b', text):
            span_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            spans.append((span_text, start_pos, end_pos))
        
        # Years (4 digits)
        for match in re.finditer(r'\b\d{4}\b', text):
            span_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            spans.append((span_text, start_pos, end_pos))
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or MM/DD/YY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY or MM-DD-YY
            r'\b\w+ \d{1,2}, \d{4}\b',       # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                span_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                spans.append((span_text, start_pos, end_pos))
        
        # Proper nouns (capitalized words)
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            span_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            spans.append((span_text, start_pos, end_pos))
        
        return spans
    
    def _extract_span_features(self, question: str, sentence: str, span: str) -> np.ndarray:
        """
        Extract features for span scoring.
        
        Args:
            question: The input question
            sentence: The sentence containing the span
            span: The candidate span
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic span properties
        span_length = len(span.split())
        features.append(min(span_length, 10))  # Cap at 10
        
        # Position features
        sentence_words = sentence.split()
        if sentence_words:
            span_words = span.split()
            try:
                span_start_idx = sentence.lower().find(span.lower())
                if span_start_idx >= 0:
                    # Relative position in sentence
                    relative_pos = span_start_idx / len(sentence)
                    features.append(relative_pos)
                else:
                    features.append(0.5)  # Default middle position
            except:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Question-span similarity
        q_span_similarity = self._compute_text_similarity(question, span)
        features.append(q_span_similarity)
        
        # Question type matching
        question_type_features = self._get_question_type_features(question, span)
        features.extend(question_type_features)
        
        # Span type features
        span_type_features = self._get_span_type_features(span)
        features.extend(span_type_features)
        
        # Context features
        context_features = self._get_context_features(sentence, span)
        features.extend(context_features)
        
        return np.array(features)
    
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
    
    def _get_question_type_features(self, question: str, span: str) -> List[float]:
        """
        Get features based on question type and span compatibility.
        
        Args:
            question: The question text
            span: The span text
            
        Returns:
            Question type compatibility features
        """
        question_lower = question.lower()
        span_lower = span.lower()
        
        # Who questions should match person names
        who_question = 1.0 if any(word in question_lower for word in ['who', 'whom', 'whose']) else 0.0
        is_person_name = 1.0 if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', span) else 0.0
        who_match = who_question * is_person_name
        
        # When questions should match dates/times
        when_question = 1.0 if 'when' in question_lower else 0.0
        is_date = 1.0 if re.search(r'\d{4}|\d{1,2}/\d{1,2}', span) else 0.0
        when_match = when_question * is_date
        
        # How many/much questions should match numbers
        how_many_question = 1.0 if any(phrase in question_lower for phrase in ['how many', 'how much']) else 0.0
        is_number = 1.0 if re.search(r'\b\d+', span) else 0.0
        how_many_match = how_many_question * is_number
        
        return [who_match, when_match, how_many_match]
    
    def _get_span_type_features(self, span: str) -> List[float]:
        """
        Get features based on span type.
        
        Args:
            span: The span text
            
        Returns:
            Span type features
        """
        # Entity type indicators
        is_person = 1.0 if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', span) else 0.0
        is_number = 1.0 if re.search(r'\b\d+', span) else 0.0
        is_date = 1.0 if re.search(r'\d{4}|\d{1,2}/\d{1,2}', span) else 0.0
        is_quoted = 1.0 if span.startswith('"') and span.endswith('"') else 0.0
        
        # Linguistic features
        is_capitalized = 1.0 if span and span[0].isupper() else 0.0
        has_articles = 1.0 if any(word in span.lower().split() for word in ['the', 'a', 'an']) else 0.0
        
        return [is_person, is_number, is_date, is_quoted, is_capitalized, has_articles]
    
    def _get_context_features(self, sentence: str, span: str) -> List[float]:
        """
        Get features based on span context within sentence.
        
        Args:
            sentence: The sentence containing the span
            span: The span text
            
        Returns:
            Context features
        """
        # Context indicators
        has_is_before = 1.0 if ' is ' in sentence.lower() and sentence.lower().find(' is ') < sentence.lower().find(span.lower()) else 0.0
        has_was_before = 1.0 if ' was ' in sentence.lower() and sentence.lower().find(' was ') < sentence.lower().find(span.lower()) else 0.0
        has_comma_after = 1.0 if span.lower() + ',' in sentence.lower() else 0.0
        
        return [has_is_before, has_was_before, has_comma_after]
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the span picker model on weakly supervised data.
        
        Args:
            training_data: List of training examples with format:
                {
                    'question': str,
                    'sentences': List[str],
                    'gold_spans': List[str],  # Gold answer spans
                }
                
        Returns:
            Training metrics
        """
        logger.info(f"Training span picker model with {len(training_data)} examples")
        
        X, y = [], []
        
        for example in training_data:
            question = example['question']
            sentences = example.get('sentences', [])
            gold_spans = set(example.get('gold_spans', []))
            
            for sentence in sentences:
                # Generate span candidates
                candidates = self._generate_span_candidates(sentence)
                
                for span_text, _, _ in candidates:
                    # Extract features
                    features = self._extract_span_features(question, sentence, span_text)
                    
                    # Create label (1 if gold span, 0 otherwise)
                    label = 1 if span_text in gold_spans else 0
                    
                    X.append(features)
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training on {len(X)} samples, {np.sum(y)} positive examples")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Compute training metrics
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'mean_positive_prob': np.mean(y_prob[y == 1]),
            'mean_negative_prob': np.mean(y_prob[y == 0])
        }
        
        logger.info(f"Span picker training completed. Metrics: {metrics}")
        return metrics
    
    def pick_best_span(self, question: str, evidence_sentences: List[str] = None, 
                      final_recall_path: Optional[str] = None) -> Tuple[str, float]:
        """
        Pick the best span from evidence sentences.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences (可选，如果提供final_recall_path则从文件读取)
            final_recall_path: final_recall.jsonl文件路径，如果提供则从此文件读取evidence
            
        Returns:
            Tuple of (best_span, score)
        """
        # 如果提供了final_recall_path，从文件读取evidence_sentences
        if final_recall_path and Path(final_recall_path).exists():
            logger.info(f"Loading evidence from final_recall_path: {final_recall_path}")
            try:
                evidence_sentences = []
                with open(final_recall_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        note = json.loads(line.strip())
                        content = note.get('content', '')
                        if content:
                            evidence_sentences.append(content)
                logger.info(f"Loaded {len(evidence_sentences)} evidence sentences from {final_recall_path}")
            except Exception as e:
                logger.error(f"Failed to load evidence from {final_recall_path}: {e}")
                if evidence_sentences is None:
                    evidence_sentences = []
        
        if not evidence_sentences:
            return "", 0.0
        
        if not self.is_trained:
            logger.warning("Model not trained, using fallback span selection")
            return self._fallback_span_selection(question, evidence_sentences)
        
        best_span = ""
        best_score = 0.0
        
        for sentence in evidence_sentences:
            # Generate span candidates
            candidates = self._generate_span_candidates(sentence)
            
            for span_text, _, _ in candidates:
                # Extract features
                features = self._extract_span_features(question, sentence, span_text)
                
                # Scale features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Predict probability
                probability = self.model.predict_proba(features_scaled)[0, 1]
                
                if probability > best_score:
                    best_score = probability
                    best_span = span_text
        
        logger.info(f"Selected span: '{best_span}' with score {best_score:.3f}")
        return best_span, best_score
    
    def _fallback_span_selection(self, question: str, evidence_sentences: List[str]) -> Tuple[str, float]:
        """
        Fallback span selection when model is not trained.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            
        Returns:
            Tuple of (best_span, score)
        """
        best_span = ""
        best_score = 0.0
        
        for sentence in evidence_sentences:
            candidates = self._generate_span_candidates(sentence)
            
            for span_text, _, _ in candidates:
                # Simple similarity-based scoring
                similarity = self._compute_text_similarity(question, span_text)
                
                if similarity > best_score:
                    best_score = similarity
                    best_span = span_text
        
        return best_span, best_score
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'min_span_length': self.min_span_length,
            'max_span_length': self.max_span_length
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Span picker model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            path: Path to load the model from
        """
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.min_span_length = model_data.get('min_span_length', 1)
        self.max_span_length = model_data.get('max_span_length', 10)
        
        logger.info(f"Span picker model loaded from {path}")
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load span picker parameters
            span_config = calibration.get('span_picker', {})
            self.min_span_length = span_config.get('min_span_length', 1)
            self.max_span_length = span_config.get('max_span_length', 10)
            
            # Load model if path is specified
            model_path = span_config.get('model_path')
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            
            logger.info(f"Span picker calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load span picker calibration: {e}")


def create_span_picker(model_type: str = "cross_encoder", calibration_path: Optional[str] = None) -> SpanPicker:
    """
    Create a SpanPicker instance.
    
    Args:
        model_type: Type of model ('cross_encoder' or 'simple')
        calibration_path: Path to calibration file
        
    Returns:
        SpanPicker instance
    """
    return SpanPicker(model_type=model_type, calibration_path=calibration_path)