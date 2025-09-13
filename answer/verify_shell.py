"""Answer verification and correction shell.

This module implements answer verification and correction using entailment scoring.
When generated answers are not trustworthy, it replaces them with original text spans
based on model scores rather than hard-coded rules.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle
import re


class AnswerVerifier:
    """Answer verification and correction system."""
    
    def __init__(self, span_picker=None, calibration_path: Optional[str] = None):
        """
        Initialize the answer verifier.
        
        Args:
            span_picker: SpanPicker instance for fallback span selection
            calibration_path: Path to calibration file
        """
        self.span_picker = span_picker
        self.calibration_path = calibration_path
        
        # Entailment model components
        self.entailment_model = None
        self.entailment_scaler = StandardScaler()
        self.is_entailment_trained = False
        
        # Thresholds (will be loaded from calibration)
        self.keep_threshold = 0.6  # τ_keep for answer retention
        self.min_evidence_overlap = 0.1  # Minimum overlap to keep generated answer
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def _extract_entailment_features(self, question: str, evidence_sentences: List[str], 
                                   answer: str) -> np.ndarray:
        """
        Extract features for entailment scoring.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            answer: The candidate answer
            
        Returns:
            Feature vector for entailment prediction
        """
        features = []
        
        # Basic answer properties
        answer_length = len(answer.split())
        features.append(min(answer_length, 20))  # Cap at 20
        
        # Question-answer similarity
        qa_similarity = self._compute_text_similarity(question, answer)
        features.append(qa_similarity)
        
        # Evidence-answer overlap
        evidence_text = " ".join(evidence_sentences)
        evidence_overlap = self._compute_evidence_overlap(answer, evidence_text)
        features.append(evidence_overlap)
        
        # Answer appears in evidence (exact match)
        exact_match = 1.0 if answer.lower() in evidence_text.lower() else 0.0
        features.append(exact_match)
        
        # Answer type consistency
        answer_type_features = self._get_answer_type_consistency(question, answer)
        features.extend(answer_type_features)
        
        # Evidence quality features
        evidence_quality_features = self._get_evidence_quality_features(evidence_sentences, answer)
        features.extend(evidence_quality_features)
        
        # Linguistic features
        linguistic_features = self._get_linguistic_features(question, answer, evidence_sentences)
        features.extend(linguistic_features)
        
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
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_evidence_overlap(self, answer: str, evidence: str) -> float:
        """
        Compute overlap between answer and evidence.
        
        Args:
            answer: The answer text
            evidence: The evidence text
            
        Returns:
            Overlap ratio
        """
        if not answer or not evidence:
            return 0.0
        
        answer_words = set(answer.lower().split())
        evidence_words = set(evidence.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(evidence_words))
        return overlap / len(answer_words)
    
    def _get_answer_type_consistency(self, question: str, answer: str) -> List[float]:
        """
        Check if answer type is consistent with question type.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Answer type consistency features
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Who questions should have person names
        who_question = 1.0 if any(word in question_lower for word in ['who', 'whom', 'whose']) else 0.0
        has_person_name = 1.0 if re.search(r'\b[A-Z][a-z]+(\s+[A-Z][a-z]+)*\b', answer) else 0.0
        who_consistency = who_question * has_person_name
        
        # When questions should have temporal expressions
        when_question = 1.0 if 'when' in question_lower else 0.0
        has_temporal = 1.0 if re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}|\bin\s+\d{4}|\bon\s+\w+', answer) else 0.0
        when_consistency = when_question * has_temporal
        
        # How many/much questions should have numbers
        how_many_question = 1.0 if any(phrase in question_lower for phrase in ['how many', 'how much']) else 0.0
        has_number = 1.0 if re.search(r'\b\d+', answer) else 0.0
        how_many_consistency = how_many_question * has_number
        
        # Where questions should have location indicators
        where_question = 1.0 if 'where' in question_lower else 0.0
        has_location = 1.0 if any(word in answer_lower for word in ['in', 'at', 'from', 'to', 'city', 'country']) else 0.0
        where_consistency = where_question * has_location
        
        return [who_consistency, when_consistency, how_many_consistency, where_consistency]
    
    def _get_evidence_quality_features(self, evidence_sentences: List[str], answer: str) -> List[float]:
        """
        Get features related to evidence quality.
        
        Args:
            evidence_sentences: List of evidence sentences
            answer: The answer text
            
        Returns:
            Evidence quality features
        """
        if not evidence_sentences:
            return [0.0, 0.0, 0.0]
        
        # Number of supporting sentences
        num_sentences = len(evidence_sentences)
        normalized_num_sentences = min(num_sentences / 5.0, 1.0)  # Normalize to [0, 1]
        
        # Average sentence length
        avg_sentence_length = np.mean([len(sent.split()) for sent in evidence_sentences])
        normalized_avg_length = min(avg_sentence_length / 30.0, 1.0)  # Normalize
        
        # Answer coverage in evidence
        answer_words = set(answer.lower().split())
        evidence_text = " ".join(evidence_sentences).lower()
        coverage = sum(1 for word in answer_words if word in evidence_text) / len(answer_words) if answer_words else 0.0
        
        return [normalized_num_sentences, normalized_avg_length, coverage]
    
    def _get_linguistic_features(self, question: str, answer: str, evidence_sentences: List[str]) -> List[float]:
        """
        Get linguistic features.
        
        Args:
            question: The question text
            answer: The answer text
            evidence_sentences: List of evidence sentences
            
        Returns:
            Linguistic features
        """
        # Answer completeness (not just a single word)
        is_complete = 1.0 if len(answer.split()) > 1 else 0.0
        
        # Answer has proper capitalization
        is_capitalized = 1.0 if answer and answer[0].isupper() else 0.0
        
        # Answer ends with punctuation
        has_punctuation = 1.0 if answer and answer[-1] in '.!?' else 0.0
        
        # Answer contains question words (might indicate incomplete processing)
        question_words = {'who', 'what', 'when', 'where', 'why', 'how'}
        has_question_words = 1.0 if any(word in answer.lower().split() for word in question_words) else 0.0
        
        return [is_complete, is_capitalized, has_punctuation, has_question_words]
    
    def train_entailment_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the entailment model for answer verification.
        
        Args:
            training_data: List of training examples with format:
                {
                    'question': str,
                    'evidence_sentences': List[str],
                    'answer': str,
                    'is_entailed': bool  # Whether answer is entailed by evidence
                }
                
        Returns:
            Training metrics
        """
        logger.info(f"Training entailment model with {len(training_data)} examples")
        
        X, y = [], []
        
        for example in training_data:
            question = example['question']
            evidence_sentences = example['evidence_sentences']
            answer = example['answer']
            is_entailed = example['is_entailed']
            
            # Extract features
            features = self._extract_entailment_features(question, evidence_sentences, answer)
            
            X.append(features)
            y.append(1 if is_entailed else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training on {len(X)} samples, {np.sum(y)} positive examples")
        
        # Scale features
        X_scaled = self.entailment_scaler.fit_transform(X)
        
        # Initialize and train model
        self.entailment_model = LogisticRegression(random_state=42, max_iter=1000)
        self.entailment_model.fit(X_scaled, y)
        self.is_entailment_trained = True
        
        # Compute training metrics
        y_pred = self.entailment_model.predict(X_scaled)
        y_prob = self.entailment_model.predict_proba(X_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'mean_positive_prob': np.mean(y_prob[y == 1]),
            'mean_negative_prob': np.mean(y_prob[y == 0])
        }
        
        logger.info(f"Entailment model training completed. Metrics: {metrics}")
        return metrics
    
    def compute_entailment_score(self, question: str, evidence_sentences: List[str], answer: str) -> float:
        """
        Compute entailment score for question-evidence-answer triplet.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            answer: The candidate answer
            
        Returns:
            Entailment score between 0 and 1
        """
        if not self.is_entailment_trained:
            logger.warning("Entailment model not trained, using fallback scoring")
            return self._fallback_entailment_scoring(question, evidence_sentences, answer)
        
        # Extract features
        features = self._extract_entailment_features(question, evidence_sentences, answer)
        
        # Scale features
        features_scaled = self.entailment_scaler.transform(features.reshape(1, -1))
        
        # Predict probability
        probability = self.entailment_model.predict_proba(features_scaled)[0, 1]
        
        return probability
    
    def _fallback_entailment_scoring(self, question: str, evidence_sentences: List[str], answer: str) -> float:
        """
        Fallback entailment scoring when model is not trained.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            answer: The candidate answer
            
        Returns:
            Simple overlap-based score
        """
        evidence_text = " ".join(evidence_sentences)
        
        # Check if answer appears in evidence
        if answer.lower() in evidence_text.lower():
            return 0.9
        
        # Otherwise use overlap
        overlap = self._compute_evidence_overlap(answer, evidence_text)
        return overlap
    
    def finalize_answer(self, question: str, raw_answer: str, evidence_sentences: List[str]) -> str:
        """
        Finalize answer using verification and correction.
        
        Args:
            question: The input question
            raw_answer: The generated answer
            evidence_sentences: List of evidence sentences
            
        Returns:
            Final answer (either original or corrected)
        """
        if not raw_answer or not evidence_sentences:
            return raw_answer
        
        logger.info(f"Verifying answer: '{raw_answer}'")
        
        # Step 1: Check if generated answer appears in evidence
        evidence_text = " ".join(evidence_sentences)
        if raw_answer.lower() in evidence_text.lower():
            logger.info("Answer found in evidence, keeping original")
            return raw_answer
        
        # Step 2: Compute entailment score
        entailment_score = self.compute_entailment_score(question, evidence_sentences, raw_answer)
        logger.info(f"Entailment score: {entailment_score:.3f}, threshold: {self.keep_threshold}")
        
        # Step 3: Decide whether to keep or replace
        if entailment_score >= self.keep_threshold:
            logger.info("Entailment score above threshold, keeping original answer")
            return raw_answer
        else:
            logger.info("Entailment score below threshold, using span picker for correction")
            return self._get_corrected_answer(question, evidence_sentences, raw_answer)
    
    def _get_corrected_answer(self, question: str, evidence_sentences: List[str], raw_answer: str) -> str:
        """
        Get corrected answer using span picker.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            raw_answer: The original generated answer
            
        Returns:
            Corrected answer
        """
        if self.span_picker:
            corrected_span, span_score = self.span_picker.pick_best_span(question, evidence_sentences)
            
            if corrected_span and span_score > 0.1:  # Minimum confidence threshold
                logger.info(f"Using corrected span: '{corrected_span}' (score: {span_score:.3f})")
                return corrected_span
        
        # Fallback: try to extract a reasonable span from evidence
        fallback_answer = self._extract_fallback_answer(question, evidence_sentences)
        if fallback_answer:
            logger.info(f"Using fallback answer: '{fallback_answer}'")
            return fallback_answer
        
        # Last resort: return original answer
        logger.warning("No suitable correction found, returning original answer")
        return raw_answer
    
    def _extract_fallback_answer(self, question: str, evidence_sentences: List[str]) -> str:
        """
        Extract fallback answer using simple heuristics.
        
        Args:
            question: The input question
            evidence_sentences: List of evidence sentences
            
        Returns:
            Fallback answer or empty string
        """
        question_lower = question.lower()
        
        for sentence in evidence_sentences:
            # For "who" questions, look for proper nouns
            if any(word in question_lower for word in ['who', 'whom']):
                names = re.findall(r'\b[A-Z][a-z]+(\s+[A-Z][a-z]+)*\b', sentence)
                if names:
                    return names[0]
            
            # For "when" questions, look for years or dates
            if 'when' in question_lower:
                dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', sentence)
                if dates:
                    return dates[0]
            
            # For "how many" questions, look for numbers
            if 'how many' in question_lower or 'how much' in question_lower:
                numbers = re.findall(r'\b\d+\b', sentence)
                if numbers:
                    return numbers[0]
        
        return ""
    
    def save_model(self, path: str) -> None:
        """
        Save the trained entailment model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_entailment_trained:
            logger.warning("No trained entailment model to save")
            return
        
        model_data = {
            'entailment_model': self.entailment_model,
            'entailment_scaler': self.entailment_scaler,
            'is_entailment_trained': self.is_entailment_trained,
            'keep_threshold': self.keep_threshold,
            'min_evidence_overlap': self.min_evidence_overlap
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Answer verifier model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained entailment model.
        
        Args:
            path: Path to load the model from
        """
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.entailment_model = model_data['entailment_model']
        self.entailment_scaler = model_data['entailment_scaler']
        self.is_entailment_trained = model_data['is_entailment_trained']
        self.keep_threshold = model_data.get('keep_threshold', 0.6)
        self.min_evidence_overlap = model_data.get('min_evidence_overlap', 0.1)
        
        logger.info(f"Answer verifier model loaded from {path}")
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load answer verification parameters
            verify_config = calibration.get('answer_verification', {})
            self.keep_threshold = verify_config.get('keep_threshold', 0.6)
            self.min_evidence_overlap = verify_config.get('min_evidence_overlap', 0.1)
            
            # Load model if path is specified
            model_path = verify_config.get('model_path')
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            
            logger.info(f"Answer verifier calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load answer verifier calibration: {e}")


def create_answer_verifier(span_picker=None, calibration_path: Optional[str] = None) -> AnswerVerifier:
    """
    Create an AnswerVerifier instance.
    
    Args:
        span_picker: SpanPicker instance for fallback
        calibration_path: Path to calibration file
        
    Returns:
        AnswerVerifier instance
    """
    return AnswerVerifier(span_picker=span_picker, calibration_path=calibration_path)