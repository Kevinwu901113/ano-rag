"""Learned fusion module for ranking paragraphs.

This module implements a learnable fusion system that replaces manual weight tuning
with a lightweight linear/MLP learner. It combines BM25 scores, vector similarities,
title-question similarities, sentence-question max similarities, and position features.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle


class LearnedFusion:
    """Learnable fusion system for paragraph ranking."""
    
    def __init__(self, model_type: str = "linear", calibration_path: Optional[str] = None):
        """
        Initialize the learned fusion system.
        
        Args:
            model_type: Type of model to use ('linear' or 'mlp')
            calibration_path: Path to calibration file
        """
        self.model_type = model_type
        self.calibration_path = calibration_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load calibration if available
        if calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
    
    def _extract_features(self, question: str, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features for each candidate paragraph.
        
        Args:
            question: The input question
            candidates: List of candidate paragraphs with metadata
            
        Returns:
            Feature matrix of shape (n_candidates, n_features)
        """
        features = []
        
        for candidate in candidates:
            feature_vector = []
            
            # BM25 score (normalized)
            bm25_score = candidate.get('bm25_score', 0.0)
            feature_vector.append(bm25_score)
            
            # Dense vector similarity
            dense_score = candidate.get('dense_score', 0.0)
            feature_vector.append(dense_score)
            
            # Title-question similarity
            title = candidate.get('title', '')
            title_sim = self._compute_text_similarity(question, title)
            feature_vector.append(title_sim)
            
            # Max sentence-question similarity within paragraph
            content = candidate.get('content', '')
            max_sent_sim = self._compute_max_sentence_similarity(question, content)
            feature_vector.append(max_sent_sim)
            
            # Position features (from packed_order if available)
            position = candidate.get('packed_position', len(candidates))
            # Normalize position to [0, 1]
            normalized_position = 1.0 - (position / len(candidates)) if len(candidates) > 0 else 0.0
            feature_vector.append(normalized_position)
            
            # Length features
            content_length = len(content.split()) if content else 0
            # Normalize length (log scale)
            log_length = np.log(content_length + 1) / 10.0  # Rough normalization
            feature_vector.append(log_length)
            
            # Graph features (if available)
            graph_score = candidate.get('graph_score', 0.0)
            feature_vector.append(graph_score)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple text similarity (can be enhanced with embeddings).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_max_sentence_similarity(self, question: str, content: str) -> float:
        """
        Compute maximum similarity between question and any sentence in content.
        
        Args:
            question: The question text
            content: The paragraph content
            
        Returns:
            Maximum sentence similarity score
        """
        if not content:
            return 0.0
        
        # Split content into sentences (simple split by periods)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        max_sim = 0.0
        for sentence in sentences:
            sim = self._compute_text_similarity(question, sentence)
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the fusion model on weakly supervised data.
        
        Args:
            training_data: List of training examples with format:
                {
                    'question': str,
                    'candidates': List[Dict],
                    'positive_ids': List[str],  # IDs of paragraphs containing gold answer
                    'negative_ids': List[str]   # IDs of other paragraphs
                }
                
        Returns:
            Training metrics
        """
        logger.info(f"Training learned fusion model with {len(training_data)} examples")
        
        X, y = [], []
        
        for example in training_data:
            question = example['question']
            candidates = example['candidates']
            positive_ids = set(example.get('positive_ids', []))
            
            # Extract features
            features = self._extract_features(question, candidates)
            
            # Create labels (1 for positive, 0 for negative)
            for i, candidate in enumerate(candidates):
                candidate_id = candidate.get('note_id', candidate.get('id', str(i)))
                label = 1 if candidate_id in positive_ids else 0
                
                X.append(features[i])
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training on {len(X)} samples, {np.sum(y)} positive examples")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == "linear":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                random_state=42,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
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
        
        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics
    
    def rank_paragraphs(self, question: str, candidates: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Rank paragraphs using the learned fusion model.
        
        Args:
            question: The input question
            candidates: List of candidate paragraphs with format:
                {
                    'id': str,
                    'title': str,
                    'content': str,
                    'bm25_score': float,
                    'dense_score': float,
                    ...
                }
                
        Returns:
            List of (paragraph_id, score) tuples sorted by score (descending)
        """
        if not self.is_trained:
            logger.warning("Model not trained, using fallback ranking")
            return self._fallback_ranking(candidates)
        
        if not candidates:
            return []
        
        # Extract features
        features = self._extract_features(question, candidates)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Create ranked results
        results = []
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get('note_id', candidate.get('id', str(i)))
            score = probabilities[i]
            results.append((candidate_id, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _fallback_ranking(self, candidates: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Fallback ranking when model is not trained.
        
        Args:
            candidates: List of candidate paragraphs
            
        Returns:
            List of (paragraph_id, score) tuples
        """
        results = []
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get('note_id', candidate.get('id', str(i)))
            # Simple linear combination as fallback
            score = (
                candidate.get('dense_score', 0.0) * 1.0 +
                candidate.get('bm25_score', 0.0) * 0.6
            )
            results.append((candidate_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
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
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
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
        
        logger.info(f"Model loaded from {path}")
    
    def _load_calibration(self, path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            path: Path to calibration file
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            # Load model if path is specified
            model_path = calibration.get('learned_fusion', {}).get('model_path')
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            
            logger.info(f"Calibration loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")


def create_learned_fusion(model_type: str = "linear", calibration_path: Optional[str] = None) -> LearnedFusion:
    """
    Create a LearnedFusion instance.
    
    Args:
        model_type: Type of model ('linear' or 'mlp')
        calibration_path: Path to calibration file
        
    Returns:
        LearnedFusion instance
    """
    return LearnedFusion(model_type=model_type, calibration_path=calibration_path)