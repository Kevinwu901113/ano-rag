#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记标准化集成模块

将实体与谓词标准化功能集成到笔记生成和处理流程中，
为原子笔记添加normalized_entities和normalized_predicates字段。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from utils.entity_predicate_normalizer import EntityPredicateNormalizer
from utils.enhanced_ner import EnhancedNER
import re

logger = logging.getLogger(__name__)

class NoteNormalizer:
    """
    笔记标准化器
    
    负责将实体与谓词标准化功能集成到笔记处理流程中，
    为笔记添加标准化的实体和谓词信息。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化实体谓词标准化器
        self.normalizer = EntityPredicateNormalizer(config)
        
        # 初始化增强NER（用于提取更多实体）
        try:
            self.enhanced_ner = EnhancedNER()
            self.use_enhanced_ner = True
        except Exception as e:
            logger.warning(f"Failed to initialize EnhancedNER: {e}")
            self.enhanced_ner = None
            self.use_enhanced_ner = False
        
        # 配置参数
        self.min_confidence = config.get('note_normalizer', {}).get('min_confidence', 0.5)
        self.enable_predicate_extraction = config.get('note_normalizer', {}).get('enable_predicate_extraction', True)
        self.enable_entity_enhancement = config.get('note_normalizer', {}).get('enable_entity_enhancement', True)
        
        # 谓词提取模式
        self.predicate_patterns = self._load_predicate_patterns()
        
        # 统计信息
        self.stats = {
            'total_notes_processed': 0,
            'entities_normalized': 0,
            'predicates_extracted': 0,
            'predicates_normalized': 0,
            'enhanced_entities_found': 0
        }
        
        logger.info(f"NoteNormalizer initialized with enhanced_ner={self.use_enhanced_ner}")
    
    def normalize_notes(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量标准化笔记
        
        Args:
            notes: 笔记列表
            
        Returns:
            标准化后的笔记列表
        """
        logger.info(f"Starting normalization for {len(notes)} notes")
        
        normalized_notes = []
        for note in notes:
            try:
                normalized_note = self.normalize_single_note(note)
                normalized_notes.append(normalized_note)
            except Exception as e:
                logger.error(f"Failed to normalize note {note.get('note_id', 'unknown')}: {e}")
                # 添加空的标准化字段，保持原始笔记
                note['normalized_entities'] = []
                note['normalized_predicates'] = []
                normalized_notes.append(note)
        
        self.stats['total_notes_processed'] += len(notes)
        logger.info(f"Completed normalization: {len(normalized_notes)} notes processed")
        
        return normalized_notes
    
    def normalize_single_note(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化单个笔记
        
        Args:
            note: 原始笔记
            
        Returns:
            标准化后的笔记
        """
        # 复制笔记以避免修改原始数据
        normalized_note = note.copy()
        
        # 获取笔记内容和现有实体
        content = note.get('content', '')
        raw_span = note.get('raw_span', content)
        existing_entities = note.get('entities', [])
        
        # 1. 增强实体提取（如果启用）
        enhanced_entities = self._extract_enhanced_entities(content, existing_entities)
        
        # 2. 标准化实体
        normalized_entities = self._normalize_entities(enhanced_entities)
        
        # 3. 提取和标准化谓词
        normalized_predicates = []
        if self.enable_predicate_extraction:
            extracted_predicates = self._extract_predicates(content, raw_span)
            normalized_predicates = self._normalize_predicates(extracted_predicates)
        
        # 4. 更新笔记
        normalized_note['normalized_entities'] = normalized_entities
        normalized_note['normalized_predicates'] = normalized_predicates
        
        # 5. 添加标准化元数据
        normalized_note['normalization_metadata'] = {
            'entity_count': len(normalized_entities),
            'predicate_count': len(normalized_predicates),
            'enhanced_entities_added': len(enhanced_entities) - len(existing_entities),
            'normalization_timestamp': self._get_timestamp()
        }
        
        return normalized_note
    
    def _extract_enhanced_entities(self, content: str, existing_entities: List[str]) -> List[str]:
        """
        使用增强NER提取更多实体
        
        Args:
            content: 文本内容
            existing_entities: 现有实体列表
            
        Returns:
            增强后的实体列表
        """
        if not self.use_enhanced_ner or not self.enable_entity_enhancement:
            return existing_entities
        
        try:
            # 使用增强NER提取实体
            enhanced_result = self.enhanced_ner.extract_entities(content)
            enhanced_entities = enhanced_result.get('entities', [])
            
            # 合并现有实体和增强实体
            all_entities = list(set(existing_entities + enhanced_entities))
            
            self.stats['enhanced_entities_found'] += len(enhanced_entities)
            
            return all_entities
            
        except Exception as e:
            logger.warning(f"Enhanced entity extraction failed: {e}")
            return existing_entities
    
    def _normalize_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        标准化实体列表
        
        Args:
            entities: 原始实体列表
            
        Returns:
            标准化实体信息列表
        """
        normalized_entities = []
        
        for entity in entities:
            if not entity or not entity.strip():
                continue
            
            try:
                # 标准化实体
                normalized_name, confidence = self.normalizer.normalize_entity(entity)
                
                # 只保留置信度足够的标准化结果
                if confidence >= self.min_confidence:
                    normalized_entities.append({
                        'original': entity,
                        'normalized': normalized_name,
                        'confidence': confidence,
                        'type': self._get_entity_type(entity)
                    })
                    self.stats['entities_normalized'] += 1
                else:
                    # 置信度不足，保留原始实体
                    normalized_entities.append({
                        'original': entity,
                        'normalized': entity,
                        'confidence': 0.0,
                        'type': self._get_entity_type(entity)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to normalize entity '{entity}': {e}")
                # 添加原始实体作为备用
                normalized_entities.append({
                    'original': entity,
                    'normalized': entity,
                    'confidence': 0.0,
                    'type': 'unknown'
                })
        
        return normalized_entities
    
    def _extract_predicates(self, content: str, raw_span: str = None) -> List[str]:
        """
        从文本中提取谓词
        
        Args:
            content: 笔记内容
            raw_span: 原始文本片段
            
        Returns:
            提取的谓词列表
        """
        predicates = []
        text_to_analyze = raw_span or content
        
        # 使用预定义模式提取谓词
        for pattern_info in self.predicate_patterns:
            pattern = pattern_info['pattern']
            predicate_type = pattern_info['type']
            
            matches = re.finditer(pattern, text_to_analyze, re.IGNORECASE)
            for match in matches:
                predicate = match.group(1) if match.groups() else match.group(0)
                predicate = predicate.strip()
                
                if predicate and len(predicate) > 1:
                    predicates.append(predicate)
        
        # 去重
        predicates = list(set(predicates))
        self.stats['predicates_extracted'] += len(predicates)
        
        return predicates
    
    def _normalize_predicates(self, predicates: List[str]) -> List[Dict[str, Any]]:
        """
        标准化谓词列表
        
        Args:
            predicates: 原始谓词列表
            
        Returns:
            标准化谓词信息列表
        """
        normalized_predicates = []
        
        for predicate in predicates:
            if not predicate or not predicate.strip():
                continue
            
            try:
                # 标准化谓词
                normalized_name, confidence = self.normalizer.normalize_predicate(predicate)
                
                # 获取谓词分类
                category = self.normalizer.get_predicate_category(normalized_name)
                
                normalized_predicates.append({
                    'original': predicate,
                    'normalized': normalized_name,
                    'confidence': confidence,
                    'category': category or 'unknown'
                })
                
                if confidence >= self.min_confidence:
                    self.stats['predicates_normalized'] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to normalize predicate '{predicate}': {e}")
                # 添加原始谓词作为备用
                normalized_predicates.append({
                    'original': predicate,
                    'normalized': predicate,
                    'confidence': 0.0,
                    'category': 'unknown'
                })
        
        return normalized_predicates
    
    def _load_predicate_patterns(self) -> List[Dict[str, Any]]:
        """
        加载谓词提取模式
        
        Returns:
            谓词提取模式列表
        """
        patterns = [
            # 中文谓词模式
            {'pattern': r'([^，。；：！？\s]+)(?:了|着|过)?\s*([^，。；：！？\s]+)', 'type': 'action'},
            {'pattern': r'([^，。；：！？\s]+)\s*(?:是|为|属于|隶属于)\s*([^，。；：！？\s]+)', 'type': 'relation'},
            {'pattern': r'([^，。；：！？\s]+)\s*(?:位于|在|坐落于|地处)\s*([^，。；：！？\s]+)', 'type': 'location'},
            {'pattern': r'([^，。；：！？\s]+)\s*(?:创立|成立|建立|创建|创办)\s*([^，。；：！？\s]+)', 'type': 'founding'},
            {'pattern': r'([^，。；：！？\s]+)\s*(?:拥有|持有|控制|管理)\s*([^，。；：！？\s]+)', 'type': 'ownership'},
            {'pattern': r'([^，。；：！？\s]+)\s*(?:担任|任职|就职于|工作于)\s*([^，。；：！？\s]+)', 'type': 'employment'},
            
            # 英文谓词模式
            {'pattern': r'([A-Za-z\s]+)\s+(?:is|was|are|were)\s+([A-Za-z\s]+)', 'type': 'relation'},
            {'pattern': r'([A-Za-z\s]+)\s+(?:founded|established|created)\s+([A-Za-z\s]+)', 'type': 'founding'},
            {'pattern': r'([A-Za-z\s]+)\s+(?:located|situated)\s+(?:in|at)\s+([A-Za-z\s]+)', 'type': 'location'},
            {'pattern': r'([A-Za-z\s]+)\s+(?:owns|controls|manages)\s+([A-Za-z\s]+)', 'type': 'ownership'},
            {'pattern': r'([A-Za-z\s]+)\s+(?:works|worked)\s+(?:at|for)\s+([A-Za-z\s]+)', 'type': 'employment'},
        ]
        
        # 从配置加载自定义模式
        custom_patterns = self.config.get('note_normalizer', {}).get('predicate_patterns', [])
        patterns.extend(custom_patterns)
        
        return patterns
    
    def _get_entity_type(self, entity: str) -> str:
        """
        简单的实体类型识别
        
        Args:
            entity: 实体名称
            
        Returns:
            实体类型
        """
        # 简单的规则匹配
        if re.search(r'(公司|集团|企业|Corp|Inc|Ltd|LLC)', entity, re.IGNORECASE):
            return 'organization'
        elif re.search(r'(市|省|县|区|国|州|City|State|Country)', entity, re.IGNORECASE):
            return 'location'
        elif re.search(r'(先生|女士|教授|博士|Mr|Ms|Dr|Prof)', entity, re.IGNORECASE):
            return 'person'
        elif re.search(r'(大学|学院|学校|University|College|School)', entity, re.IGNORECASE):
            return 'educational_institution'
        else:
            return 'unknown'
    
    def _get_timestamp(self) -> float:
        """
        获取当前时间戳
        
        Returns:
            时间戳
        """
        import time
        return time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """
        重置统计信息
        """
        for key in self.stats:
            self.stats[key] = 0
    
    def add_entity_alias(self, canonical_name: str, alias: str, confidence: float = 1.0) -> None:
        """
        添加实体别名
        
        Args:
            canonical_name: 标准名称
            alias: 别名
            confidence: 置信度
        """
        self.normalizer.add_entity_alias(canonical_name, alias, confidence)
    
    def add_predicate_mapping(self, original: str, normalized: str) -> None:
        """
        添加谓词映射
        
        Args:
            original: 原始谓词
            normalized: 标准化谓词
        """
        self.normalizer.add_predicate_mapping(original, normalized)
    
    def save_alias_dict(self, file_path: str) -> None:
        """
        保存别名词典
        
        Args:
            file_path: 保存路径
        """
        self.normalizer.save_alias_dict(file_path)


# 便利函数
def create_note_normalizer(config: Dict[str, Any]) -> NoteNormalizer:
    """
    创建笔记标准化器实例
    
    Args:
        config: 配置字典
        
    Returns:
        笔记标准化器实例
    """
    return NoteNormalizer(config)


def normalize_notes_batch(notes: List[Dict[str, Any]], 
                         config: Dict[str, Any],
                         normalizer: Optional[NoteNormalizer] = None) -> List[Dict[str, Any]]:
    """
    批量标准化笔记的便利函数
    
    Args:
        notes: 笔记列表
        config: 配置字典
        normalizer: 可选的标准化器实例
        
    Returns:
        标准化后的笔记列表
    """
    if normalizer is None:
        normalizer = create_note_normalizer(config)
    
    return normalizer.normalize_notes(notes)