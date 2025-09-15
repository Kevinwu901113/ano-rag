from typing import List, Dict, Any, Optional, Set, Tuple
import re
import numpy as np
from collections import Counter, defaultdict
from loguru import logger
import spacy
from datetime import datetime
import dateutil.parser as date_parser

class AtomicNoteFeatureExtractor:
    """
    原子笔记特征提取器
    计算候选笔记的多维特征，包括：
    - 命中事实数
    - 命中事实的平均重要性
    - 谓词/时间覆盖度
    - 跨句多样性
    """
    
    def __init__(self):
        # 尝试加载spacy模型，如果失败则使用基础方法
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            logger.warning("spaCy model not found, using basic text processing")
            self.nlp = None
            self.use_spacy = False
        
        # 预定义的重要性权重
        self.importance_weights = {
            'entity': 1.0,
            'predicate': 0.8,
            'temporal': 0.9,
            'numerical': 0.7,
            'location': 0.8
        }
        
        # 时间表达式模式
        self.temporal_patterns = [
            r'\b\d{4}\b',  # 年份
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # 日期
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\b',
            r'\b(today|yesterday|tomorrow|now|recently|currently)\b',
            r'\b\d+\s+(years?|months?|days?|hours?|minutes?)\s+(ago|later)\b'
        ]
        
        # 数值表达式模式
        self.numerical_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|hundred)?\b',
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            r'\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b'
        ]
    
    def extract_features(self, candidates: List[Dict[str, Any]], query: str, 
                        query_entities: List[str] = None, 
                        query_predicates: List[str] = None) -> List[Dict[str, Any]]:
        """
        为候选笔记提取原子笔记特征
        
        Args:
            candidates: 候选笔记列表
            query: 查询文本
            query_entities: 查询中的实体列表
            query_predicates: 查询中的谓词列表
            
        Returns:
            增强了特征的候选笔记列表
        """
        if not candidates:
            return candidates
        
        # 提取查询特征
        if query_entities is None:
            query_entities = self._extract_entities(query)
        if query_predicates is None:
            query_predicates = self._extract_predicates(query)
        
        query_temporal = self._extract_temporal_expressions(query)
        query_numerical = self._extract_numerical_expressions(query)
        
        enhanced_candidates = []
        
        for candidate in candidates:
            # 复制原始候选
            enhanced_candidate = candidate.copy()
            
            # 提取候选笔记的基本信息
            content = candidate.get('content', '')
            title = candidate.get('title', '')
            full_text = f"{title} {content}"
            
            # 计算各种特征
            features = self._calculate_atomic_features(
                full_text, query, query_entities, query_predicates, 
                query_temporal, query_numerical
            )
            
            # 将特征添加到候选中
            enhanced_candidate.update(features)
            enhanced_candidates.append(enhanced_candidate)
        
        return enhanced_candidates
    
    def _calculate_atomic_features(self, text: str, query: str, 
                                 query_entities: List[str], 
                                 query_predicates: List[str],
                                 query_temporal: List[str],
                                 query_numerical: List[str]) -> Dict[str, float]:
        """
        计算单个文本的原子笔记特征
        """
        features = {}
        
        # 1. 命中事实数特征
        entity_hits = self._count_entity_hits(text, query_entities)
        predicate_hits = self._count_predicate_hits(text, query_predicates)
        temporal_hits = self._count_temporal_hits(text, query_temporal)
        numerical_hits = self._count_numerical_hits(text, query_numerical)
        
        total_fact_hits = entity_hits + predicate_hits + temporal_hits + numerical_hits
        features['fact_hit_count'] = total_fact_hits
        features['entity_hit_count'] = entity_hits
        features['predicate_hit_count'] = predicate_hits
        features['temporal_hit_count'] = temporal_hits
        features['numerical_hit_count'] = numerical_hits
        
        # 2. 命中事实的平均重要性
        importance_score = self._calculate_importance_score(
            entity_hits, predicate_hits, temporal_hits, numerical_hits
        )
        features['avg_fact_importance'] = importance_score
        
        # 3. 谓词/时间覆盖度
        predicate_coverage = self._calculate_predicate_coverage(text, query_predicates)
        temporal_coverage = self._calculate_temporal_coverage(text, query_temporal)
        features['predicate_coverage'] = predicate_coverage
        features['temporal_coverage'] = temporal_coverage
        features['combined_coverage'] = (predicate_coverage + temporal_coverage) / 2
        
        # 4. 跨句多样性
        sentence_diversity = self._calculate_sentence_diversity(text, query)
        features['sentence_diversity'] = sentence_diversity
        
        # 5. 文本质量特征
        text_quality = self._calculate_text_quality(text)
        features.update(text_quality)
        
        # 6. 语义密度特征
        semantic_density = self._calculate_semantic_density(text, query)
        features['semantic_density'] = semantic_density
        
        return features
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
        else:
            # 基础实体提取：大写开头的词组
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities = [e.lower() for e in entities]
        
        return list(set(entities))
    
    def _extract_predicates(self, text: str) -> List[str]:
        """提取谓词（动词短语）"""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            predicates = [token.lemma_.lower() for token in doc 
                         if token.pos_ in ['VERB'] and len(token.text) > 2]
        else:
            # 基础谓词提取：常见动词模式
            verb_patterns = r'\b(?:is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|may|might)\s+\w+\b'
            predicates = re.findall(verb_patterns, text.lower())
        
        return list(set(predicates))
    
    def _extract_temporal_expressions(self, text: str) -> List[str]:
        """提取时间表达式"""
        temporal_exprs = []
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_exprs.extend(matches)
        return list(set([expr.lower() for expr in temporal_exprs]))
    
    def _extract_numerical_expressions(self, text: str) -> List[str]:
        """提取数值表达式"""
        numerical_exprs = []
        for pattern in self.numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numerical_exprs.extend(matches)
        return list(set([expr.lower() for expr in numerical_exprs]))
    
    def _count_entity_hits(self, text: str, query_entities: List[str]) -> int:
        """计算实体命中数"""
        if not query_entities:
            return 0
        
        text_lower = text.lower()
        hits = 0
        for entity in query_entities:
            if entity.lower() in text_lower:
                hits += 1
        return hits
    
    def _count_predicate_hits(self, text: str, query_predicates: List[str]) -> int:
        """计算谓词命中数"""
        if not query_predicates:
            return 0
        
        text_lower = text.lower()
        hits = 0
        for predicate in query_predicates:
            if predicate.lower() in text_lower:
                hits += 1
        return hits
    
    def _count_temporal_hits(self, text: str, query_temporal: List[str]) -> int:
        """计算时间表达式命中数"""
        if not query_temporal:
            return 0
        
        text_lower = text.lower()
        hits = 0
        for temporal in query_temporal:
            if temporal.lower() in text_lower:
                hits += 1
        return hits
    
    def _count_numerical_hits(self, text: str, query_numerical: List[str]) -> int:
        """计算数值表达式命中数"""
        if not query_numerical:
            return 0
        
        text_lower = text.lower()
        hits = 0
        for numerical in query_numerical:
            if numerical.lower() in text_lower:
                hits += 1
        return hits
    
    def _calculate_importance_score(self, entity_hits: int, predicate_hits: int, 
                                  temporal_hits: int, numerical_hits: int) -> float:
        """计算平均重要性分数"""
        total_hits = entity_hits + predicate_hits + temporal_hits + numerical_hits
        if total_hits == 0:
            return 0.0
        
        weighted_score = (
            entity_hits * self.importance_weights['entity'] +
            predicate_hits * self.importance_weights['predicate'] +
            temporal_hits * self.importance_weights['temporal'] +
            numerical_hits * self.importance_weights['numerical']
        )
        
        return weighted_score / total_hits
    
    def _calculate_predicate_coverage(self, text: str, query_predicates: List[str]) -> float:
        """计算谓词覆盖度"""
        if not query_predicates:
            return 0.0
        
        text_lower = text.lower()
        covered_predicates = sum(1 for pred in query_predicates 
                               if pred.lower() in text_lower)
        
        return covered_predicates / len(query_predicates)
    
    def _calculate_temporal_coverage(self, text: str, query_temporal: List[str]) -> float:
        """计算时间覆盖度"""
        if not query_temporal:
            return 0.0
        
        text_lower = text.lower()
        covered_temporal = sum(1 for temp in query_temporal 
                             if temp.lower() in text_lower)
        
        return covered_temporal / len(query_temporal)
    
    def _calculate_sentence_diversity(self, text: str, query: str) -> float:
        """计算跨句多样性"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.0
        
        query_words = set(query.lower().split())
        sentence_overlaps = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            sentence_overlaps.append(overlap)
        
        # 计算句子间的多样性（标准差）
        if len(sentence_overlaps) > 1:
            diversity = np.std(sentence_overlaps) / (np.mean(sentence_overlaps) + 1e-6)
        else:
            diversity = 0.0
        
        return min(diversity, 1.0)  # 归一化到[0,1]
    
    def _calculate_text_quality(self, text: str) -> Dict[str, float]:
        """计算文本质量特征"""
        features = {}
        
        # 文本长度特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # 词汇丰富度
        words = text.lower().split()
        unique_words = set(words)
        features['lexical_diversity'] = len(unique_words) / max(len(words), 1)
        
        # 平均句长
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            features['avg_sentence_length'] = avg_sentence_length
        else:
            features['avg_sentence_length'] = 0.0
        
        return features
    
    def _calculate_semantic_density(self, text: str, query: str) -> float:
        """计算语义密度"""
        text_words = set(text.lower().split())
        query_words = set(query.lower().split())
        
        if not text_words:
            return 0.0
        
        # 计算查询词在文本中的密度
        common_words = text_words.intersection(query_words)
        density = len(common_words) / len(text_words)
        
        return density
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return [
            'fact_hit_count', 'entity_hit_count', 'predicate_hit_count', 
            'temporal_hit_count', 'numerical_hit_count', 'avg_fact_importance',
            'predicate_coverage', 'temporal_coverage', 'combined_coverage',
            'sentence_diversity', 'text_length', 'word_count', 'sentence_count',
            'lexical_diversity', 'avg_sentence_length', 'semantic_density'
        ]
    
    def normalize_features(self, candidates: List[Dict[str, Any]], 
                         feature_names: List[str] = None) -> List[Dict[str, Any]]:
        """归一化特征值到[0,1]范围"""
        if not candidates:
            return candidates
        
        if feature_names is None:
            feature_names = self.get_feature_names()
        
        # 计算每个特征的最大值和最小值
        feature_stats = {}
        for feature_name in feature_names:
            values = [c.get(feature_name, 0.0) for c in candidates]
            if values:
                feature_stats[feature_name] = {
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values)
                }
        
        # 归一化特征
        normalized_candidates = []
        for candidate in candidates:
            normalized_candidate = candidate.copy()
            
            for feature_name in feature_names:
                if feature_name in feature_stats:
                    stats = feature_stats[feature_name]
                    value = candidate.get(feature_name, 0.0)
                    
                    if stats['range'] > 0:
                        normalized_value = (value - stats['min']) / stats['range']
                    else:
                        normalized_value = 0.0
                    
                    normalized_candidate[f'{feature_name}_norm'] = normalized_value
            
            normalized_candidates.append(normalized_candidate)
        
        return normalized_candidates