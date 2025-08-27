"""实体到笔记的倒排索引，支持图优先的二跳检索策略"""

import json
import os
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from loguru import logger
import pickle
from utils.file_utils import FileUtils
from config import config

class EntityInvertedIndex:
    """实体到笔记的倒排索引
    
    支持:
    1. 从桥接实体快速获取相关笔记候选集合
    2. 实体名称标准化和别名处理
    3. 索引的持久化存储和加载
    4. 增量更新支持
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        # 核心索引结构：entity -> set of note_ids
        self.entity_to_notes: Dict[str, Set[str]] = defaultdict(set)
        
        # 实体别名映射：alias -> canonical_entity
        self.entity_aliases: Dict[str, str] = {}
        
        # 笔记到实体的反向索引：note_id -> set of entities
        self.note_to_entities: Dict[str, Set[str]] = defaultdict(set)
        
        # 实体统计信息
        self.entity_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 存储路径
        if storage_path is None:
            storage_path = config.get('storage.entity_index_path', 'data/entity_inverted_index')
        self.storage_path = storage_path
        FileUtils.ensure_dir(os.path.dirname(self.storage_path))
        
        # 配置参数
        self.min_entity_length = config.get('entity_index.min_entity_length', 2)
        self.max_entity_length = config.get('entity_index.max_entity_length', 100)
        self.enable_fuzzy_matching = config.get('entity_index.enable_fuzzy_matching', True)
        
        logger.info(f"EntityInvertedIndex initialized with storage path: {self.storage_path}")
    
    def build_index(self, atomic_notes: List[Dict[str, Any]]) -> None:
        """从原子笔记构建倒排索引"""
        logger.info(f"Building entity inverted index from {len(atomic_notes)} atomic notes")
        
        # 清空现有索引
        self.entity_to_notes.clear()
        self.entity_aliases.clear()
        self.note_to_entities.clear()
        self.entity_stats.clear()
        
        processed_count = 0
        for note in atomic_notes:
            try:
                self._index_note(note)
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(atomic_notes)} notes")
                    
            except Exception as e:
                logger.warning(f"Failed to index note {note.get('note_id', 'unknown')}: {e}")
        
        # 计算统计信息
        self._compute_statistics()
        
        logger.info(f"Entity inverted index built: {len(self.entity_to_notes)} entities, "
                   f"{len(self.note_to_entities)} notes indexed")
    
    def _index_note(self, note: Dict[str, Any]) -> None:
        """为单个笔记建立索引"""
        note_id = note.get('note_id')
        if not note_id:
            return
        
        # 从笔记中提取实体
        entities = self._extract_entities_from_note(note)
        
        for entity in entities:
            # 标准化实体名称
            canonical_entity = self._normalize_entity(entity)
            
            if self._is_valid_entity(canonical_entity):
                # 更新倒排索引
                self.entity_to_notes[canonical_entity].add(note_id)
                self.note_to_entities[note_id].add(canonical_entity)
                
                # 如果原始实体名与标准化名称不同，建立别名映射
                if entity != canonical_entity:
                    self.entity_aliases[entity] = canonical_entity
    
    def _extract_entities_from_note(self, note: Dict[str, Any]) -> List[str]:
        """从笔记中提取实体"""
        entities = []
        
        # 从entities字段提取
        if 'entities' in note and isinstance(note['entities'], list):
            entities.extend(note['entities'])
        
        # 从relations字段提取
        if 'relations' in note and isinstance(note['relations'], list):
            for relation in note['relations']:
                if isinstance(relation, dict):
                    if 'subject' in relation:
                        entities.append(relation['subject'])
                    if 'object' in relation:
                        entities.append(relation['object'])
        
        # 从raw_span_evidence字段提取（如果存在）
        if 'raw_span_evidence' in note:
            evidence_entities = self._extract_entities_from_evidence(note['raw_span_evidence'])
            entities.extend(evidence_entities)
        
        return list(set(entities))  # 去重
    
    def _extract_entities_from_evidence(self, evidence: str) -> List[str]:
        """从证据句中提取实体
        
        解析类似 "A co founded B" 这样的简单证据句
        """
        entities = []
        
        # 简单的实体抽取规则
        # 匹配常见的关系模式
        import re
        
        # 匹配 "A relation B" 模式
        patterns = [
            r'([A-Z][\w\s]+?)\s+(co founded|founded|located in|member of|works for|part of|instance of)\s+([A-Z][\w\s]+)',
            r'([A-Z][\w\s]+?)\s+(is|was|are|were)\s+([A-Z][\w\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, evidence, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    entities.append(match[0].strip())
                    entities.append(match[2].strip())
                elif len(match) >= 2:
                    entities.append(match[0].strip())
                    entities.append(match[1].strip())
        
        return entities
    
    def _normalize_entity(self, entity: str) -> str:
        """标准化实体名称"""
        if not isinstance(entity, str):
            return str(entity)
        
        # 基本清理
        normalized = entity.strip()
        
        # 移除多余的空格
        normalized = ' '.join(normalized.split())
        
        # 首字母大写（保持其他字母的原始大小写）
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]
        
        return normalized
    
    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        if not entity or not isinstance(entity, str):
            return False
        
        # 长度检查
        if len(entity) < self.min_entity_length or len(entity) > self.max_entity_length:
            return False
        
        # 过滤常见的无效实体
        invalid_entities = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if entity.lower() in invalid_entities:
            return False
        
        return True
    
    def get_candidate_notes(self, bridge_entities: List[str]) -> Set[str]:
        """根据桥接实体列表获取候选笔记集合"""
        candidate_notes = set()
        
        for entity in bridge_entities:
            # 标准化实体名称
            canonical_entity = self._normalize_entity(entity)
            
            # 直接查找
            if canonical_entity in self.entity_to_notes:
                candidate_notes.update(self.entity_to_notes[canonical_entity])
            
            # 查找别名
            if entity in self.entity_aliases:
                canonical_entity = self.entity_aliases[entity]
                if canonical_entity in self.entity_to_notes:
                    candidate_notes.update(self.entity_to_notes[canonical_entity])
            
            # 模糊匹配（如果启用）
            if self.enable_fuzzy_matching and not candidate_notes:
                fuzzy_matches = self._fuzzy_match_entity(entity)
                for match in fuzzy_matches:
                    candidate_notes.update(self.entity_to_notes[match])
        
        return candidate_notes
    
    def _fuzzy_match_entity(self, entity: str, threshold: float = 0.8) -> List[str]:
        """模糊匹配实体名称"""
        from difflib import SequenceMatcher
        
        matches = []
        entity_lower = entity.lower()
        
        for indexed_entity in self.entity_to_notes.keys():
            similarity = SequenceMatcher(None, entity_lower, indexed_entity.lower()).ratio()
            if similarity >= threshold:
                matches.append(indexed_entity)
        
        return matches
    
    def _compute_statistics(self) -> None:
        """计算索引统计信息"""
        for entity, note_ids in self.entity_to_notes.items():
            self.entity_stats[entity] = {
                'note_count': len(note_ids),
                'frequency': len(note_ids)  # 可以后续扩展为更复杂的频率计算
            }
    
    def get_entity_statistics(self, entity: str) -> Dict[str, Any]:
        """获取实体统计信息"""
        canonical_entity = self._normalize_entity(entity)
        
        # 检查别名
        if entity in self.entity_aliases:
            canonical_entity = self.entity_aliases[entity]
        
        return self.entity_stats.get(canonical_entity, {})
    
    def add_note(self, note: Dict[str, Any]) -> None:
        """增量添加笔记到索引"""
        try:
            self._index_note(note)
            logger.debug(f"Added note {note.get('note_id')} to entity index")
        except Exception as e:
            logger.warning(f"Failed to add note to entity index: {e}")
    
    def remove_note(self, note_id: str) -> None:
        """从索引中移除笔记"""
        if note_id in self.note_to_entities:
            entities = self.note_to_entities[note_id].copy()
            
            for entity in entities:
                if entity in self.entity_to_notes:
                    self.entity_to_notes[entity].discard(note_id)
                    
                    # 如果该实体不再有关联笔记，移除实体
                    if not self.entity_to_notes[entity]:
                        del self.entity_to_notes[entity]
                        if entity in self.entity_stats:
                            del self.entity_stats[entity]
            
            del self.note_to_entities[note_id]
            logger.debug(f"Removed note {note_id} from entity index")
    
    def save_index(self) -> None:
        """保存索引到磁盘"""
        try:
            index_data = {
                'entity_to_notes': {k: list(v) for k, v in self.entity_to_notes.items()},
                'entity_aliases': self.entity_aliases,
                'note_to_entities': {k: list(v) for k, v in self.note_to_entities.items()},
                'entity_stats': self.entity_stats
            }
            
            with open(f"{self.storage_path}.pkl", 'wb') as f:
                pickle.dump(index_data, f)
            
            # 同时保存JSON格式便于调试
            with open(f"{self.storage_path}.json", 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Entity inverted index saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save entity index: {e}")
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            pkl_path = f"{self.storage_path}.pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.entity_to_notes = defaultdict(set)
                for k, v in index_data['entity_to_notes'].items():
                    self.entity_to_notes[k] = set(v)
                
                self.entity_aliases = index_data.get('entity_aliases', {})
                
                self.note_to_entities = defaultdict(set)
                for k, v in index_data['note_to_entities'].items():
                    self.note_to_entities[k] = set(v)
                
                self.entity_stats = index_data.get('entity_stats', {})
                
                logger.info(f"Entity inverted index loaded from {pkl_path}")
                logger.info(f"Loaded {len(self.entity_to_notes)} entities, {len(self.note_to_entities)} notes")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load entity index: {e}")
        
        return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        return {
            'total_entities': len(self.entity_to_notes),
            'total_notes': len(self.note_to_entities),
            'total_aliases': len(self.entity_aliases),
            'avg_notes_per_entity': sum(len(notes) for notes in self.entity_to_notes.values()) / max(1, len(self.entity_to_notes)),
            'avg_entities_per_note': sum(len(entities) for entities in self.note_to_entities.values()) / max(1, len(self.note_to_entities))
        }