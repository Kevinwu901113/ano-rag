from typing import List, Dict, Any, Tuple, Set
import numpy as np
from loguru import logger
from dataclasses import dataclass
import re
from collections import defaultdict

from utils.text_utils import TextUtils
from graph.relation_extractor import RelationExtractor
from llm.local_llm import LocalLLM

@dataclass
class EnhancedRelation:
    """增强的关系数据结构"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    reasoning_value: float  # 推理价值分数
    metadata: Dict[str, Any]
    
class EnhancedRelationExtractor(RelationExtractor):
    """增强的关系提取器，专门针对多跳推理优化"""
    
    def __init__(self, llm: LocalLLM = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
        # 新增关系类型
        self.enhanced_relation_types = {
            'causal': {'weight': 0.9, 'description': '因果关系'},
            'temporal': {'weight': 0.8, 'description': '时序关系'},
            'comparison': {'weight': 0.7, 'description': '比较关系'},
            'definition': {'weight': 0.8, 'description': '定义关系'},
            'part_whole': {'weight': 0.7, 'description': '部分-整体关系'},
            'instance': {'weight': 0.6, 'description': '实例关系'},
            'contradiction': {'weight': 0.8, 'description': '矛盾关系'},
            'support': {'weight': 0.7, 'description': '支持关系'},
            'reference': {'weight': 0.5, 'description': '引用关系'},
            'entity_cooccurrence': {'weight': 0.4, 'description': '实体共现关系'},
            'contextual': {'weight': 0.6, 'description': '上下文关系'},
            'semantic_similarity': {'weight': 0.5, 'description': '语义相似性关系'},
            'reasoning_path': {'weight': 1.0, 'description': '推理路径关系'}
        }
        
        # 推理模式
        self.reasoning_patterns = {
            'qa_pattern': r'问题.*?答案|问.*?答|Q.*?A',
            'definition_pattern': r'定义|是指|即|也就是说|换句话说',
            'causal_pattern': r'因为|由于|导致|造成|引起|结果|所以|因此'
        }
        
    def extract_relations(self, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """提取增强关系"""
        logger.info(f"开始提取增强关系，共{len(notes)}个笔记")
        
        relations = []
        
        # 1. 提取传统关系（继承原有功能）
        traditional_relations = self._extract_traditional_relations(notes)
        relations.extend(traditional_relations)
        
        # 2. 提取推理路径关系
        reasoning_relations = self._extract_reasoning_path_relations(notes)
        relations.extend(reasoning_relations)
        
        # 3. 使用LLM提取语义关系（如果可用）
        if self.llm:
            semantic_relations = self._extract_semantic_relations_with_llm(notes)
            relations.extend(semantic_relations)
        
        # 4. 关系去重和过滤
        relations = self._deduplicate_and_filter_relations(relations)
        
        # 5. 计算推理价值
        relations = self._calculate_reasoning_values(relations)
        
        logger.info(f"提取完成，共{len(relations)}个增强关系")
        return relations
    
    def _extract_traditional_relations(self, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """提取传统关系（继承原有功能）"""
        relations = []
        
        for i, note1 in enumerate(notes):
            for j, note2 in enumerate(notes[i+1:], i+1):
                # 引用关系
                if self._has_reference_relation(note1, note2):
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='reference',
                        confidence=0.8,
                        reasoning_value=0.0,  # 稍后计算
                        metadata={'type': 'traditional'}
                    ))
                
                # 实体共现关系
                entity_score = self._calculate_entity_cooccurrence(note1, note2)
                if entity_score > 0.3:
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='entity_cooccurrence',
                        confidence=entity_score,
                        reasoning_value=0.0,
                        metadata={'score': entity_score, 'type': 'traditional'}
                    ))
                
                # 语义相似性关系
                similarity_score = self._calculate_text_similarity(note1['content'], note2['content'])
                if similarity_score > 0.4:  # 降低阈值以捕获更多关系
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='semantic_similarity',
                        confidence=similarity_score,
                        reasoning_value=0.0,
                        metadata={'score': similarity_score, 'type': 'traditional'}
                    ))
        
        return relations
    
    def _extract_reasoning_path_relations(self, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """提取推理路径关系"""
        relations = []
        
        for i, note1 in enumerate(notes):
            for j, note2 in enumerate(notes[i+1:], i+1):
                # 检查问答模式
                if self._matches_qa_pattern(note1, note2):
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='reasoning_path',
                        confidence=0.9,
                        reasoning_value=0.0,
                        metadata={'pattern': 'qa', 'type': 'reasoning'}
                    ))
                
                # 检查定义模式
                if self._matches_definition_pattern(note1, note2):
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='definition',
                        confidence=0.8,
                        reasoning_value=0.0,
                        metadata={'pattern': 'definition', 'type': 'reasoning'}
                    ))
                
                # 检查因果模式
                if self._matches_causal_pattern(note1, note2):
                    relations.append(EnhancedRelation(
                        source_id=note1['id'],
                        target_id=note2['id'],
                        relation_type='causal',
                        confidence=0.85,
                        reasoning_value=0.0,
                        metadata={'pattern': 'causal', 'type': 'reasoning'}
                    ))
        
        return relations
    
    def _extract_semantic_relations_with_llm(self, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """使用LLM提取语义关系"""
        relations = []
        
        # 批量处理以提高效率
        batch_size = 5
        for i in range(0, len(notes), batch_size):
            batch_notes = notes[i:i+batch_size]
            batch_relations = self._extract_batch_semantic_relations(batch_notes)
            relations.extend(batch_relations)
        
        return relations
    
    def _extract_batch_semantic_relations(self, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """批量提取语义关系"""
        if not self.llm:
            return []
        
        # 构建提示
        notes_text = "\n\n".join([f"笔记{i+1}: {note['content'][:200]}" for i, note in enumerate(notes)])
        
        prompt = f"""
分析以下笔记之间的语义关系，识别可能的推理路径：

{notes_text}

请识别笔记之间的关系类型：
1. causal（因果关系）
2. temporal（时序关系）
3. comparison（比较关系）
4. definition（定义关系）
5. part_whole（部分-整体关系）
6. instance（实例关系）
7. contradiction（矛盾关系）
8. support（支持关系）

输出格式：
笔记X -> 笔记Y: 关系类型 (置信度0-1)
"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=500)
            return self._parse_llm_relations(response, notes)
        except Exception as e:
            logger.warning(f"LLM关系提取失败: {e}")
            return []
    
    def _parse_llm_relations(self, response: str, notes: List[Dict[str, Any]]) -> List[EnhancedRelation]:
        """解析LLM输出的关系"""
        relations = []
        
        # 解析格式：笔记X -> 笔记Y: 关系类型 (置信度)
        pattern = r'笔记(\d+)\s*->\s*笔记(\d+):\s*(\w+)\s*\((\d*\.?\d+)\)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            try:
                source_idx = int(match[0]) - 1
                target_idx = int(match[1]) - 1
                relation_type = match[2]
                confidence = float(match[3])
                
                if (0 <= source_idx < len(notes) and 
                    0 <= target_idx < len(notes) and 
                    relation_type in self.enhanced_relation_types):
                    
                    relations.append(EnhancedRelation(
                        source_id=notes[source_idx]['id'],
                        target_id=notes[target_idx]['id'],
                        relation_type=relation_type,
                        confidence=confidence,
                        reasoning_value=0.0,
                        metadata={'source': 'llm', 'type': 'semantic'}
                    ))
            except (ValueError, IndexError) as e:
                logger.warning(f"解析LLM关系失败: {e}")
                continue
        
        return relations
    
    def _deduplicate_and_filter_relations(self, relations: List[EnhancedRelation]) -> List[EnhancedRelation]:
        """关系去重和过滤"""
        # 按关系键去重
        relation_dict = {}
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type)
            if key not in relation_dict or relation.confidence > relation_dict[key].confidence:
                relation_dict[key] = relation
        
        # 过滤低置信度关系
        filtered_relations = [
            rel for rel in relation_dict.values() 
            if rel.confidence >= 0.3
        ]
        
        return filtered_relations
    
    def _calculate_reasoning_values(self, relations: List[EnhancedRelation]) -> List[EnhancedRelation]:
        """计算推理价值"""
        for relation in relations:
            # 基础推理价值 = 关系类型权重 * 置信度
            base_value = self.enhanced_relation_types[relation.relation_type]['weight'] * relation.confidence
            
            # 根据关系类型调整
            if relation.relation_type in ['causal', 'reasoning_path', 'definition']:
                base_value *= 1.2  # 提升推理关键关系的价值
            elif relation.relation_type in ['contradiction', 'comparison']:
                base_value *= 1.1  # 提升对比分析关系的价值
            
            relation.reasoning_value = min(base_value, 1.0)
        
        return relations
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 使用TextUtils的关键词相似度计算方法
        return TextUtils.calculate_similarity_keywords(text1, text2)
    
    def _has_reference_relation(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> bool:
        """检查是否存在引用关系"""
        # 简化的引用检测
        content1 = note1.get('content', '').lower()
        content2 = note2.get('content', '').lower()
        
        # 检查是否有明显的引用词汇
        reference_keywords = ['参考', '引用', '来源', '根据', '如前所述', '上述', '前面提到']
        
        for keyword in reference_keywords:
            if keyword in content1 or keyword in content2:
                return True
        
        return False
    
    def _calculate_entity_cooccurrence(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> float:
        """计算实体共现分数"""
        entities1 = set(note1.get('entities', []))
        entities2 = set(note2.get('entities', []))
        
        if not entities1 or not entities2:
            return 0.0
        
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _matches_qa_pattern(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> bool:
        """检查是否匹配问答模式"""
        content1 = note1.get('content', '')
        content2 = note2.get('content', '')
        
        pattern = self.reasoning_patterns['qa_pattern']
        return bool(re.search(pattern, content1 + content2, re.IGNORECASE))
    
    def _matches_definition_pattern(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> bool:
        """检查是否匹配定义模式"""
        content1 = note1.get('content', '')
        content2 = note2.get('content', '')
        
        pattern = self.reasoning_patterns['definition_pattern']
        return bool(re.search(pattern, content1 + content2, re.IGNORECASE))
    
    def _matches_causal_pattern(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> bool:
        """检查是否匹配因果模式"""
        content1 = note1.get('content', '')
        content2 = note2.get('content', '')
        
        pattern = self.reasoning_patterns['causal_pattern']
        return bool(re.search(pattern, content1 + content2, re.IGNORECASE))
    
    def get_relation_statistics(self, relations: List[EnhancedRelation]) -> Dict[str, Any]:
        """获取关系统计信息"""
        stats = {
            'total_relations': len(relations),
            'relation_types': defaultdict(int),
            'avg_confidence': 0.0,
            'avg_reasoning_value': 0.0
        }
        
        if not relations:
            return stats
        
        total_confidence = 0.0
        total_reasoning_value = 0.0
        
        for relation in relations:
            stats['relation_types'][relation.relation_type] += 1
            total_confidence += relation.confidence
            total_reasoning_value += relation.reasoning_value
        
        stats['avg_confidence'] = total_confidence / len(relations)
        stats['avg_reasoning_value'] = total_reasoning_value / len(relations)
        
        return dict(stats)