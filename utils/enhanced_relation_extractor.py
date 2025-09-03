from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from graph.relation_extractor import RelationExtractor
from utils.enhanced_ner import EnhancedNER
from utils.text_utils import TextUtils
from config import config
import re

class EnhancedRelationExtractor:
    """增强的关系抽取器，基于RelationExtractor扩展更多语义关系抽取能力"""
    
    def __init__(self, local_llm=None):
        # 基础关系抽取器
        self.base_extractor = RelationExtractor(local_llm=local_llm)
        
        # 增强NER组件
        self.enhanced_ner = EnhancedNER()
        
        # 配置
        self.config = config.get('enhanced_relation_extraction', {})
        self.enable_semantic_relations = self.config.get('enable_semantic_relations', True)
        self.enable_entity_linking = self.config.get('enable_entity_linking', True)
        
        # 语义关系模式
        self.semantic_patterns = {
            'causal': [
                r'因为.*所以', r'由于.*导致', r'造成', r'引起', r'导致',
                r'because.*therefore', r'due to.*result', r'cause', r'lead to'
            ],
            'temporal': [
                r'之前', r'之后', r'同时', r'接着', r'然后', r'最后',
                r'before', r'after', r'meanwhile', r'then', r'finally'
            ],
            'definition': [
                r'是指', r'定义为', r'即', r'也就是说',
                r'is defined as', r'refers to', r'means', r'i.e.'
            ],
            'comparison': [
                r'相比', r'与.*不同', r'类似于', r'比.*更',
                r'compared to', r'unlike', r'similar to', r'more than'
            ]
        }
        
        logger.info("EnhancedRelationExtractor initialized")
    
    def extract_relations_from_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从原子笔记中抽取增强关系"""
        logger.info(f"Extracting enhanced relations from {len(atomic_notes)} notes")
        
        # 使用基础抽取器获取基础关系
        enhanced_notes = []
        
        for note in atomic_notes:
            enhanced_note = note.copy()
            
            # 抽取基础关系
            base_relations = self._extract_base_relations(note)
            
            # 抽取语义关系
            semantic_relations = self._extract_semantic_relations(note) if self.enable_semantic_relations else []
            
            # 合并关系
            all_relations = base_relations + semantic_relations
            enhanced_note['extracted_relations'] = all_relations
            
            enhanced_notes.append(enhanced_note)
        
        logger.info(f"Enhanced relation extraction completed")
        return enhanced_notes
    
    def _extract_base_relations(self, note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """抽取基础关系"""
        relations = []
        content = note.get('content', '')
        entities = note.get('entities', [])
        
        # 确保entities是列表
        if not isinstance(entities, list):
            logger.warning(f"entities is not a list, got {type(entities)}: {entities}")
            return relations
        
        # 实体间关系
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relation_type = self._identify_entity_relation(entity1, entity2, content)
                if relation_type:
                    relations.append({
                        'source': entity1,
                        'target': entity2,
                        'type': relation_type,
                        'confidence': 0.8,
                        'context': content[:200]
                    })
        
        return relations
    
    def _extract_semantic_relations(self, note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """抽取语义关系"""
        relations = []
        content = note.get('content', '')
        
        for relation_type, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    relations.append({
                        'type': relation_type,
                        'pattern': pattern,
                        'match': match.group(),
                        'position': match.span(),
                        'confidence': 0.7
                    })
        
        return relations
    
    def _identify_entity_relation(self, entity1: str, entity2: str, context: str) -> Optional[str]:
        """识别两个实体之间的关系类型"""
        context_lower = context.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # 查找实体在文本中的位置
        pos1 = context_lower.find(entity1_lower)
        pos2 = context_lower.find(entity2_lower)
        
        if pos1 == -1 or pos2 == -1:
            return None
        
        # 获取实体间的文本
        start_pos = min(pos1, pos2)
        end_pos = max(pos1 + len(entity1_lower), pos2 + len(entity2_lower))
        between_text = context_lower[start_pos:end_pos]
        
        # 基于关键词识别关系类型
        if any(word in between_text for word in ['的', 'of', 'belongs to']):
            return 'ownership'
        elif any(word in between_text for word in ['创建', '建立', 'created', 'established']):
            return 'creation'
        elif any(word in between_text for word in ['位于', '在', 'located', 'in']):
            return 'location'
        elif any(word in between_text for word in ['是', 'is', 'are']):
            return 'identity'
        
        return 'related'
    
    def build_relation_graph(self, enhanced_notes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建关系图"""
        graph = {}
        
        for note in enhanced_notes:
            note_id = note.get('note_id', '')
            relations = note.get('extracted_relations', [])
            
            if note_id not in graph:
                graph[note_id] = []
            
            for relation in relations:
                if 'target' in relation:
                    target = relation['target']
                    if target not in graph[note_id]:
                        graph[note_id].append(target)
        
        return graph
    
    def get_relation_statistics(self, enhanced_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取关系抽取统计信息"""
        total_relations = 0
        relation_types = {}
        notes_with_relations = 0
        
        for note in enhanced_notes:
            relations = note.get('extracted_relations', [])
            if relations:
                notes_with_relations += 1
                total_relations += len(relations)
                
                for relation in relations:
                    rel_type = relation.get('type', 'unknown')
                    relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        return {
            'total_relations': total_relations,
            'notes_with_relations': notes_with_relations,
            'relation_types': relation_types,
            'average_relations_per_note': total_relations / len(enhanced_notes) if enhanced_notes else 0
        }