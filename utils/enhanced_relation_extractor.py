from typing import List, Dict, Any, Set, Tuple, Optional
import re
from loguru import logger
from collections import defaultdict

class EnhancedRelationExtractor:
    """增强的关系抽取模块，基于规则模板和语义分析"""
    
    def __init__(self):
        # 关系模板规则
        self.relation_patterns = {
            'spouse': [
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:wife|husband|spouse)\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:and|&)\s+(\w+(?:\s+\w+)*?)\s+(?:are\s+)?married',
                r'(\w+(?:\s+\w+)*?)\s+married\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:wife|husband)\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+and\s+(?:his|her)\s+(?:wife|husband)\s+(\w+(?:\s+\w+)*)',
            ],
            'actor_of': [
                r'(\w+(?:\s+\w+)*?)\s+(?:voices?|voiced)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?voice\s+(?:actor\s+)?(?:of\s+|for\s+)(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:plays?|played)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:portrays?|portrayed)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:voice\s+)?(?:actor\s+)?(?:behind|for)\s+(\w+(?:\s+\w+)*)',
            ],
            'creator_of': [
                r'(\w+(?:\s+\w+)*?)\s+(?:created|creates)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?creator\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:founded|established)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:developed|designed)\s+(\w+(?:\s+\w+)*)',
            ],
            'associated_with': [
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:known\s+for|famous\s+for)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:works?\s+(?:on|with)|worked\s+(?:on|with))\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:appears?\s+in|appeared\s+in)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:part\s+of|member\s+of)\s+(\w+(?:\s+\w+)*)',
            ],
            'family_relation': [
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:son|daughter|child)\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:father|mother|parent)\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:brother|sister|sibling)\s+of\s+(\w+(?:\s+\w+)*)',
            ],
            'work_relation': [
                r'(\w+(?:\s+\w+)*?)\s+(?:works?\s+(?:for|at)|worked\s+(?:for|at))\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:employed\s+(?:by|at))\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:the\s+)?(?:CEO|director|manager|president)\s+of\s+(\w+(?:\s+\w+)*)',
            ],
            'location_relation': [
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:located\s+in|from|born\s+in)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:lives?\s+in|lived\s+in)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*?)\s+(?:is\s+)?(?:based\s+in)\s+(\w+(?:\s+\w+)*)',
            ]
        }
        
        # 关系权重
        self.relation_weights = {
            'spouse': 1.0,
            'actor_of': 0.9,
            'creator_of': 0.8,
            'family_relation': 0.9,
            'work_relation': 0.7,
            'location_relation': 0.6,
            'associated_with': 0.5
        }
        
        # 实体类型识别
        self.entity_type_patterns = {
            'person': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 标准姓名
                r'\b(?:Mr|Mrs|Ms|Dr|Professor)\s+[A-Z][a-z]+\b',  # 带称谓的姓名
            ],
            'character': [
                r'\b(?:character|role)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:character|role)\b',
            ],
            'organization': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Company|Corporation|Inc|Ltd|LLC)\b',
                r'\b(?:Company|Corporation|Inc|Ltd|LLC)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            ],
            'location': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:City|State|Country|Province)\b',
                r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            ]
        }
    
    def extract_relations_from_text(self, text: str, entities: List[str] = None) -> List[Dict[str, Any]]:
        """从文本中提取关系"""
        relations = []
        
        # 如果没有提供实体列表，先提取实体
        if entities is None:
            from .enhanced_ner import EnhancedNER
            ner = EnhancedNER()
            entities = ner.extract_entities(text)
        
        # 使用规则模板提取关系
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity1 = match.group(1).strip()
                    entity2 = match.group(2).strip()
                    
                    # 验证实体
                    if self._is_valid_relation_entity(entity1) and self._is_valid_relation_entity(entity2):
                        relation = {
                            'source_entity': entity1,
                            'target_entity': entity2,
                            'relation_type': relation_type,
                            'confidence': self.relation_weights.get(relation_type, 0.5),
                            'evidence_text': match.group(0),
                            'extraction_method': 'rule_based'
                        }
                        relations.append(relation)
        
        # 去重和过滤
        relations = self._deduplicate_relations(relations)
        
        return relations
    
    def extract_relations_from_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从原子笔记中提取关系"""
        logger.info(f"Extracting relations from {len(atomic_notes)} atomic notes")
        
        all_relations = []
        
        for note in atomic_notes:
            content = note.get('content', '')
            entities = note.get('entities', [])
            note_id = note.get('note_id', '')
            
            # 提取关系
            relations = self.extract_relations_from_text(content, entities)
            
            # 添加笔记信息
            for relation in relations:
                relation['source_note_id'] = note_id
                relation['source_content'] = content
                all_relations.append(relation)
        
        logger.info(f"Extracted {len(all_relations)} relations")
        return all_relations
    
    def enhance_atomic_notes_with_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为原子笔记添加关系信息"""
        logger.info("Enhancing atomic notes with relation information")
        
        # 提取所有关系
        all_relations = self.extract_relations_from_notes(atomic_notes)
        
        # 按笔记ID分组关系
        relations_by_note = defaultdict(list)
        for relation in all_relations:
            note_id = relation['source_note_id']
            relations_by_note[note_id].append(relation)
        
        # 为每个笔记添加关系信息
        enhanced_notes = []
        for note in atomic_notes:
            enhanced_note = note.copy()
            note_id = note.get('note_id', '')
            
            # 添加关系信息
            note_relations = relations_by_note.get(note_id, [])
            enhanced_note['entity_relations'] = note_relations
            enhanced_note['relation_count'] = len(note_relations)
            
            # 统计关系类型
            relation_types = defaultdict(int)
            for relation in note_relations:
                relation_types[relation['relation_type']] += 1
            enhanced_note['relation_types'] = dict(relation_types)
            
            enhanced_notes.append(enhanced_note)
        
        logger.info(f"Enhanced {len(enhanced_notes)} notes with relation information")
        return enhanced_notes
    
    def build_relation_graph(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建关系图"""
        logger.info("Building relation graph from atomic notes")
        
        # 提取所有关系
        all_relations = self.extract_relations_from_notes(atomic_notes)
        
        # 构建节点（实体）
        entities = set()
        for relation in all_relations:
            entities.add(relation['source_entity'])
            entities.add(relation['target_entity'])
        
        # 构建边（关系）
        edges = []
        for relation in all_relations:
            edge = {
                'source': relation['source_entity'],
                'target': relation['target_entity'],
                'relation_type': relation['relation_type'],
                'weight': relation['confidence'],
                'evidence': relation['evidence_text'],
                'source_note_id': relation['source_note_id']
            }
            edges.append(edge)
        
        # 统计信息
        relation_stats = defaultdict(int)
        for relation in all_relations:
            relation_stats[relation['relation_type']] += 1
        
        graph = {
            'nodes': list(entities),
            'edges': edges,
            'node_count': len(entities),
            'edge_count': len(edges),
            'relation_statistics': dict(relation_stats),
            'created_at': self._get_timestamp()
        }
        
        logger.info(f"Built relation graph with {len(entities)} nodes and {len(edges)} edges")
        return graph
    
    def _is_valid_relation_entity(self, entity: str) -> bool:
        """验证关系实体是否有效"""
        if not entity or len(entity.strip()) < 2:
            return False
        
        # 过滤明显的非实体
        invalid_entities = {
            'and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'he', 'she', 'it', 'they', 'we', 'you', 'i'
        }
        
        if entity.lower().strip() in invalid_entities:
            return False
        
        # 检查是否包含字母
        if not re.search(r'[a-zA-Z]', entity):
            return False
        
        return True
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 创建关系的唯一标识
            key = (
                relation['source_entity'].lower(),
                relation['target_entity'].lower(),
                relation['relation_type']
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # 如果已存在，保留置信度更高的
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing['source_entity'].lower(),
                        existing['target_entity'].lower(),
                        existing['relation_type']
                    )
                    if existing_key == key and relation['confidence'] > existing['confidence']:
                        unique_relations[i] = relation
                        break
        
        return unique_relations
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def extract_contextual_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取上下文关系（基于实体共现和语义相似性）"""
        logger.info("Extracting contextual relations between notes")
        
        contextual_relations = []
        
        for i, note1 in enumerate(atomic_notes):
            for j, note2 in enumerate(atomic_notes[i+1:], i+1):
                # 检查实体共现
                entities1 = set(note1.get('entities', []))
                entities2 = set(note2.get('entities', []))
                common_entities = entities1.intersection(entities2)
                
                if common_entities:
                    # 计算关系强度
                    strength = len(common_entities) / len(entities1.union(entities2))
                    
                    relation = {
                        'source_note_id': note1.get('note_id', ''),
                        'target_note_id': note2.get('note_id', ''),
                        'relation_type': 'entity_cooccurrence',
                        'common_entities': list(common_entities),
                        'strength': strength,
                        'confidence': min(strength * 2, 1.0),  # 转换为置信度
                        'extraction_method': 'contextual'
                    }
                    contextual_relations.append(relation)
        
        logger.info(f"Extracted {len(contextual_relations)} contextual relations")
        return contextual_relations
    
    def get_relation_statistics(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取关系统计信息"""
        if not relations:
            return {'total_relations': 0}
        
        stats = {
            'total_relations': len(relations),
            'relation_types': defaultdict(int),
            'extraction_methods': defaultdict(int),
            'confidence_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0  # < 0.5
            },
            'average_confidence': 0.0
        }
        
        total_confidence = 0
        for relation in relations:
            # 统计关系类型
            stats['relation_types'][relation.get('relation_type', 'unknown')] += 1
            
            # 统计提取方法
            stats['extraction_methods'][relation.get('extraction_method', 'unknown')] += 1
            
            # 统计置信度分布
            confidence = relation.get('confidence', 0.0)
            total_confidence += confidence
            
            if confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        stats['average_confidence'] = total_confidence / len(relations)
        stats['relation_types'] = dict(stats['relation_types'])
        stats['extraction_methods'] = dict(stats['extraction_methods'])
        
        return stats