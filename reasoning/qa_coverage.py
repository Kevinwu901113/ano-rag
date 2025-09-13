"""QA Coverage module for sentence-level question answering assessment.

This module implements sentence-level QA coverage scoring to replace relation word rules.
It uses cross-encoder/NLI models to determine if a sentence can directly answer a sub-question.
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


class QACoverageScorer:
    """Sentence-level QA coverage scoring system."""
    
    def __init__(self, model_type: str = "cross_encoder", calibration_path: Optional[str] = None):
        """
        Initialize the QA coverage scorer.
        
        Args:
            model_type: Type of model to use ('cross_encoder', 'nli', or 'simple')
            calibration_path: Path to calibration file
        """
        self.model_type = model_type
        self.calibration_path = calibration_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.threshold = 0.5
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def _extract_qa_features(self, question: str, sentence: str) -> np.ndarray:
        """
        Extract features for QA coverage scoring.
        
        Args:
            question: The sub-question
            sentence: The candidate sentence
            
        Returns:
            Feature vector
        """
        features = []
        
        # Text similarity features
        word_overlap = self._compute_word_overlap(question, sentence)
        features.append(word_overlap)
        
        # Question type features
        question_type = self._identify_question_type(question)
        features.extend(question_type)
        
        # Answer pattern features
        answer_patterns = self._detect_answer_patterns(sentence)
        features.extend(answer_patterns)
        
        # Length and structure features
        length_ratio = len(sentence.split()) / max(len(question.split()), 1)
        features.append(min(length_ratio, 5.0))  # Cap at 5.0
        
        # Syntactic features
        syntactic_features = self._extract_syntactic_features(question, sentence)
        features.extend(syntactic_features)
        
        return np.array(features)
    
    def _compute_word_overlap(self, question: str, sentence: str) -> float:
        """
        Compute word overlap between question and sentence.
        
        Args:
            question: The question text
            sentence: The sentence text
            
        Returns:
            Word overlap ratio
        """
        if not question or not sentence:
            return 0.0
        
        q_words = set(question.lower().split())
        s_words = set(sentence.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        q_words = q_words - stop_words
        s_words = s_words - stop_words
        
        if not q_words or not s_words:
            return 0.0
        
        intersection = len(q_words.intersection(s_words))
        union = len(q_words.union(s_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_question_type(self, question: str) -> List[float]:
        """
        Identify question type features.
        
        Args:
            question: The question text
            
        Returns:
            One-hot encoded question type features
        """
        question_lower = question.lower()
        
        # Question type indicators
        who_question = 1.0 if any(word in question_lower for word in ['who', 'whom', 'whose']) else 0.0
        what_question = 1.0 if 'what' in question_lower else 0.0
        when_question = 1.0 if 'when' in question_lower else 0.0
        where_question = 1.0 if 'where' in question_lower else 0.0
        why_question = 1.0 if 'why' in question_lower else 0.0
        how_question = 1.0 if any(word in question_lower for word in ['how', 'how many', 'how much']) else 0.0
        
        return [who_question, what_question, when_question, where_question, why_question, how_question]
    
    def _detect_answer_patterns(self, sentence: str) -> List[float]:
        """
        Detect answer patterns in the sentence.
        
        Args:
            sentence: The sentence text
            
        Returns:
            Answer pattern features
        """
        sentence_lower = sentence.lower()
        
        # Named entity patterns
        has_person = 1.0 if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence) else 0.0
        has_date = 1.0 if re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', sentence) else 0.0
        has_number = 1.0 if re.search(r'\b\d+\b', sentence) else 0.0
        has_location = 1.0 if any(word in sentence_lower for word in ['in', 'at', 'from', 'to', 'city', 'country', 'state']) else 0.0
        
        # Answer indicators
        has_is_verb = 1.0 if any(word in sentence_lower for word in [' is ', ' was ', ' are ', ' were ']) else 0.0
        has_performed = 1.0 if any(word in sentence_lower for word in ['performed', 'sang', 'played', 'acted', 'directed']) else 0.0
        has_relationship = 1.0 if any(word in sentence_lower for word in ['married', 'spouse', 'husband', 'wife', 'partner']) else 0.0
        
        return [has_person, has_date, has_number, has_location, has_is_verb, has_performed, has_relationship]
    
    def _extract_syntactic_features(self, question: str, sentence: str) -> List[float]:
        """
        Extract syntactic features.
        
        Args:
            question: The question text
            sentence: The sentence text
            
        Returns:
            Syntactic features
        """
        # Simple syntactic features
        q_has_question_mark = 1.0 if '?' in question else 0.0
        s_has_comma = 1.0 if ',' in sentence else 0.0
        s_has_parentheses = 1.0 if '(' in sentence and ')' in sentence else 0.0
        s_has_quotes = 1.0 if '"' in sentence or "'" in sentence else 0.0
        
        return [q_has_question_mark, s_has_comma, s_has_parentheses, s_has_quotes]
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the QA coverage model on weakly supervised data.
        
        Args:
            training_data: List of training examples with format:
                {
                    'question': str,
                    'sentences': List[str],
                    'positive_sentences': List[str],  # Sentences that can answer the question
                    'negative_sentences': List[str]   # Sentences that cannot answer
                }
                
        Returns:
            Training metrics
        """
        logger.info(f"Training QA coverage model with {len(training_data)} examples")
        
        X, y = [], []
        
        for example in training_data:
            question = example['question']
            positive_sentences = set(example.get('positive_sentences', []))
            sentences = example.get('sentences', [])
            
            for sentence in sentences:
                # Extract features
                features = self._extract_qa_features(question, sentence)
                
                # Create label (1 if can answer, 0 if cannot)
                label = 1 if sentence in positive_sentences else 0
                
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
        
        logger.info(f"QA coverage training completed. Metrics: {metrics}")
        return metrics
    
    def score_sentence(self, question: str, sentence: str) -> float:
        """
        Score how well a sentence can answer a question.
        
        Args:
            question: The input question
            sentence: The candidate sentence
            
        Returns:
            Score between 0 and 1 (higher means better coverage)
        """
        if not self.is_trained:
            logger.warning("Model not trained, using fallback scoring")
            return self._fallback_scoring(question, sentence)
        
        # Extract features
        features = self._extract_qa_features(question, sentence)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict probability
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        return probability
    
    def best_answering_sentence(self, question: str, paragraph_text: str) -> Tuple[str, float]:
        """
        Find the best sentence in a paragraph for answering a question.
        
        Args:
            question: The input question
            paragraph_text: The paragraph content
            
        Returns:
            Tuple of (best_sentence, score)
        """
        if not paragraph_text:
            return "", 0.0
        
        # Split paragraph into sentences
        sentences = self._split_sentences(paragraph_text)
        
        if not sentences:
            return "", 0.0
        
        best_sentence = ""
        best_score = 0.0
        
        for sentence in sentences:
            score = self.score_sentence(question, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence, best_score
    
    def best_answering_paragraph(self, question: str, candidates: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Find the best paragraph for answering a question.
        
        Args:
            question: The input question
            candidates: List of candidate paragraphs with 'id' and 'content' fields
            
        Returns:
            Tuple of (paragraph_id, score)
        """
        if not candidates:
            return "", 0.0
        
        best_id = ""
        best_score = 0.0
        
        for candidate in candidates:
            paragraph_id = candidate.get('id', candidate.get('note_id', ''))
            content = candidate.get('content', '')
            
            # Get best sentence score for this paragraph
            _, score = self.best_answering_sentence(question, content)
            
            if score > best_score:
                best_score = score
                best_id = paragraph_id
        
        return best_id, best_score
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with NLTK or spaCy)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences
    
    def _fallback_scoring(self, question: str, sentence: str) -> float:
        """
        Fallback scoring when model is not trained.
        
        Args:
            question: The question text
            sentence: The sentence text
            
        Returns:
            Simple overlap-based score
        """
        return self._compute_word_overlap(question, sentence)
    
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
            'threshold': self.threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"QA coverage model saved to {path}")
    
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
        self.threshold = model_data.get('threshold', 0.5)
        
        logger.info(f"QA coverage model loaded from {path}")
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load QA coverage parameters
            qa_config = calibration.get('qa_coverage', {})
            self.threshold = qa_config.get('threshold', 0.5)
            
            # Load model if path is specified
            model_path = qa_config.get('model_path')
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            
            logger.info(f"QA coverage calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load QA coverage calibration: {e}")


def create_qa_coverage_scorer(model_type: str = "cross_encoder", calibration_path: Optional[str] = None) -> QACoverageScorer:
    """
    Create a QACoverageScorer instance.
    
    Args:
        model_type: Type of model ('cross_encoder', 'nli', or 'simple')
        calibration_path: Path to calibration file
        
    Returns:
        QACoverageScorer instance
    """
    return QACoverageScorer(model_type=model_type, calibration_path=calibration_path)