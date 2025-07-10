import re
import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
from utils import TextUtils, GPUUtils, BatchProcessor
from config import config
from llm import LocalLLM

class EnhancedRelationExtractor:
    """增强的关系提取器，专门针对多跳推理优化"""
    
    def __init__(self):
        # 基础配置
        self.similarity_threshold = config.get('graph.similarity_threshold', 0.6)  # 降低阈值
        self.entity_cooccurrence_threshold = config.get('graph.entity_cooccurrence_threshold', 2)
        self.context_window = config.get('graph.context_window', 3)
        self.max_relations_per_note = config.get('graph.max_relations_per_note', 15)  # 增加关系数量
        
        # 多跳推理配置
        self.multi_hop_config = config.get('multi_hop', {})
        self.max_reasoning_hops = self.multi_hop_config.get('max_reasoning_hops', 3)
        self.min_path_confidence = self.multi_hop_config.get('min_path_confidence', 0.6)
        
        # LLM关系提取配置
        self.llm_extraction_enabled = self.multi_hop_config.get('llm_relation_extraction', {}).get('enabled', True)
        self.llm = LocalLLM() if self.llm_extraction_enabled else None
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=config.get('graph.batch_size', 64),
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # 扩展的关系类型和权重
        self.relation_types = {
            'reference': {'weight': 1.0, 'reasoning_value': 0.9},
            'entity_coexistence': {'weight': 0.8, 'reasoning_value': 0.7},
            'context_relation': {'weight': 0.6, 'reasoning_value': 0.6},
            'topic_relation': {'weight': 0.7, 'reasoning_value': 0.5},
            'semantic_similarity': {'weight': 0.5, 'reasoning_value': 0.4},
            # 新增的推理关系类型
            'causal': {'weight': 1.2, 'reasoning_value': 1.0},
            'temporal': {'weight': 1.1, 'reasoning_value': 0.9},
            'comparison': {'weight': 0.9, 'reasoning_value': 0.8},
            'definition': {'weight': 1.3, 'reasoning_value': 1.0},
            'part_of': {'weight': 1.0, 'reasoning_value': 0.8},
            'instance_of': {'weight': 1.1, 'reasoning_value': 0.9},
            'contradiction': {'weight': 0.8, 'reasoning_value': 0.7},
            'support': {'weight': 0.9, 'reasoning_value': 0.8}
        }
        
        logger.info("EnhancedRelationExtractor initialized with multi-hop reasoning support")
    
    def extract_all_relations(self, atomic_notes: List[Dict[str, Any]], 
                             embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """提取所有类型的关系，包括推理关系"""
        if not atomic_notes:
            return []
        
        logger.info(f"Extracting enhanced relations from {len(atomic_notes)} atomic notes")
        
        all_relations = []
        
        # 1. 基础关系提取
        logger.info("Extracting basic relations")
        basic_relations = self._extract_basic_relations(atomic_notes, embeddings)
        all_relations.extend(basic_relations)
        
        # 2. 智能语义关系提取（使用LLM）
        if self.llm_extraction_enabled and self.llm:
            logger.info("Extracting semantic relations with LLM")
            semantic_relations = self._extract_semantic_relations_with_llm(atomic_notes)
            all_relations.extend(semantic_relations)
        
        # 3. 推理路径关系提取
        logger.info("Extracting reasoning path relations")
        reasoning_relations = self._extract_reasoning_path_relations(atomic_notes)
        all_relations.extend(reasoning_relations)
        
        # 去重和过滤
        filtered_relations = self._filter_and_deduplicate_relations(all_relations)
        
        # 计算推理价值得分
        for relation in filtered_relations:
            relation['reasoning_value'] = self._calculate_reasoning_value(relation)
        
        logger.info(f"Extracted {len(filtered_relations)} enhanced relations ({len(all_relations)} before filtering)")
        return filtered_relations
    
    def _extract_basic_relations(self, atomic_notes: List[Dict[str, Any]], 
                               embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """提取基础关系"""
        relations = []
        
        # 引用关系
        relations.extend(self._extract_reference_relations(atomic_notes))
        
        # 实体共存关系
        relations.extend(self._extract_entity_coexistence_relations(atomic_notes))
        
        # 上下文关系
        relations.extend(self._extract_context_relations(atomic_notes))
        
        # 主题关系
        relations.extend(self._extract_topic_relations(atomic_notes))
        
        # 语义相似性关系
        if embeddings is not None:
            relations.extend(self._extract_semantic_similarity_relations(atomic_notes, embeddings))
        
        return relations
    
    def _extract_semantic_relations_with_llm(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用LLM提取语义关系"""
        relations = []
        
        # 选择候选笔记对进行分析
        candidate_pairs = self._select_candidate_pairs_for_llm_analysis(atomic_notes)
        
        logger.info(f"Analyzing {len(candidate_pairs)} note pairs with LLM")
        
        # 批量处理
        batch_size = self.multi_hop_config.get('llm_relation_extraction', {}).get('batch_size', 16)
        
        for i in range(0, len(candidate_pairs), batch_size):
            batch_pairs = candidate_pairs[i:i + batch_size]
            batch_relations = self._analyze_relations_batch(batch_pairs)
            relations.extend(batch_relations)
        
        return relations
    
    def _select_candidate_pairs_for_llm_analysis(self, atomic_notes: List[Dict[str, Any]]) -> List[Tuple[Dict, Dict]]:
        """选择候选笔记对进行LLM分析"""
        pairs = []
        max_pairs = self.multi_hop_config.get('llm_relation_extraction', {}).get('max_pairs_per_batch', 50)
        
        # 基于关键词和实体重叠选择候选对
        for i, note1 in enumerate(atomic_notes):
            for j, note2 in enumerate(atomic_notes[i+1:], i+1):
                if len(pairs) >= max_pairs:
                    break
                
                # 计算初步相关性
                relevance_score = self._calculate_pair_relevance(note1, note2)
                
                if relevance_score > 0.3:  # 只分析有一定相关性的对
                    pairs.append((note1, note2))
        
        # 按相关性排序，选择最有希望的对
        pairs.sort(key=lambda x: self._calculate_pair_relevance(x[0], x[1]), reverse=True)
        return pairs[:max_pairs]
    
    def _calculate_pair_relevance(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> float:
        """计算笔记对的初步相关性"""
        # 关键词重叠
        keywords1 = set(note1.get('keywords', []))
        keywords2 = set(note2.get('keywords', []))
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
        
        # 实体重叠
        entities1 = set(note1.get('entities', []))
        entities2 = set(note2.get('entities', []))
        entity_overlap = len(entities1 & entities2) / max(len(entities1 | entities2), 1)
        
        # 文本长度相似性（避免过短或过长的文本）
        len1, len2 = len(note1.get('content', '')), len(note2.get('content', ''))
        length_similarity = 1 - abs(len1 - len2) / max(len1 + len2, 1)
        
        return 0.4 * keyword_overlap + 0.4 * entity_overlap + 0.2 * length_similarity
    
    def _analyze_relations_batch(self, note_pairs: List[Tuple[Dict, Dict]]) -> List[Dict[str, Any]]:
        """批量分析笔记对的关系"""
        relations = []
        
        for note1, note2 in note_pairs:
            try:
                relation = self._analyze_single_pair_relation(note1, note2)
                if relation:
                    relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to analyze relation between {note1.get('note_id')} and {note2.get('note_id')}: {e}")
        
        return relations
    
    def _analyze_single_pair_relation(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析单个笔记对的关系"""
        content1 = note1.get('content', '')[:500]  # 限制长度
        content2 = note2.get('content', '')[:500]
        
        prompt = f"""
分析以下两个文本片段之间的语义关系：

文本1：{content1}

文本2：{content2}

请识别它们之间的关系类型和强度。可能的关系类型包括：
- causal（因果关系）
- temporal（时序关系）
- comparison（比较关系）
- definition（定义关系）
- part_of（部分-整体关系）
- instance_of（实例关系）
- contradiction（矛盾关系）
- support（支持关系）
- none（无明显关系）

请返回JSON格式：
{{
    "relation_type": "关系类型",
    "strength": 0.8,
    "confidence": 0.9,
    "reasoning": "关系判断的理由",
    "direction": "bidirectional/forward/backward"
}}

如果没有明显关系，请返回relation_type为"none"。
"""
        
        try:
            response = self.llm.generate(prompt)
            relation_info = self._parse_llm_relation_response(response)
            
            if (relation_info and 
                relation_info.get('relation_type') != 'none' and 
                relation_info.get('confidence', 0) > self.min_path_confidence):
                
                return {
                    'source_id': note1.get('note_id'),
                    'target_id': note2.get('note_id'),
                    'relation_type': relation_info['relation_type'],
                    'weight': relation_info.get('strength', 0.5),
                    'metadata': {
                        'reasoning': relation_info.get('reasoning', ''),
                        'confidence': relation_info.get('confidence', 0.5),
                        'direction': relation_info.get('direction', 'bidirectional'),
                        'extraction_method': 'llm_semantic'
                    }
                }
        except Exception as e:
            logger.warning(f"LLM relation analysis failed: {e}")
        
        return None
    
    def _parse_llm_relation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM关系分析响应"""
        try:
            # 清理响应
            cleaned_response = self._clean_json_response(response)
            if not cleaned_response:
                return None
            
            relation_info = json.loads(cleaned_response)
            
            # 验证必要字段
            required_fields = ['relation_type', 'strength', 'confidence']
            if not all(field in relation_info for field in required_fields):
                return None
            
            # 验证关系类型
            valid_types = list(self.relation_types.keys()) + ['none']
            if relation_info['relation_type'] not in valid_types:
                return None
            
            return relation_info
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM relation response: {e}")
            return None
    
    def _clean_json_response(self, response: str) -> str:
        """清理LLM响应中的JSON"""
        if not response:
            return ""
        
        # 移除markdown代码块
        import re
        response = re.sub(r'```(?:json)?\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # 查找JSON对象
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx+1]
        
        return ""
    
    def _extract_reasoning_path_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取推理路径关系"""
        relations = []
        
        # 基于问答模式的关系
        qa_relations = self._extract_qa_pattern_relations(atomic_notes)
        relations.extend(qa_relations)
        
        # 基于定义模式的关系
        definition_relations = self._extract_definition_pattern_relations(atomic_notes)
        relations.extend(definition_relations)
        
        # 基于因果模式的关系
        causal_relations = self._extract_causal_pattern_relations(atomic_notes)
        relations.extend(causal_relations)
        
        return relations
    
    def _extract_qa_pattern_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取问答模式关系"""
        relations = []
        
        # 识别问题和答案模式
        question_patterns = [
            r'什么是(.+?)\?',
            r'(.+?)是什么',
            r'如何(.+?)\?',
            r'为什么(.+?)\?',
            r'哪里(.+?)\?',
            r'什么时候(.+?)\?'
        ]
        
        answer_patterns = [
            r'(.+?)是(.+)',
            r'(.+?)的定义是(.+)',
            r'(.+?)指的是(.+)',
            r'(.+?)表示(.+)'
        ]
        
        for i, note1 in enumerate(atomic_notes):
            content1 = note1.get('content', '')
            
            # 检查是否包含问题模式
            is_question = any(re.search(pattern, content1) for pattern in question_patterns)
            
            if is_question:
                # 寻找可能的答案
                for j, note2 in enumerate(atomic_notes):
                    if i == j:
                        continue
                    
                    content2 = note2.get('content', '')
                    
                    # 检查是否包含答案模式
                    is_answer = any(re.search(pattern, content2) for pattern in answer_patterns)
                    
                    if is_answer:
                        # 计算问答相关性
                        relevance = self._calculate_qa_relevance(content1, content2)
                        
                        if relevance > 0.6:
                            relations.append({
                                'source_id': note1.get('note_id'),
                                'target_id': note2.get('note_id'),
                                'relation_type': 'definition',
                                'weight': relevance * self.relation_types['definition']['weight'],
                                'metadata': {
                                    'pattern_type': 'qa_pattern',
                                    'relevance_score': relevance
                                }
                            })
        
        return relations
    
    def _calculate_qa_relevance(self, question: str, answer: str) -> float:
        """计算问答相关性"""
        # 使用TextUtils的关键词相似度计算方法
        return TextUtils.calculate_similarity_keywords(question, answer)
    
    def _extract_definition_pattern_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取定义模式关系"""
        relations = []
        
        definition_patterns = [
            r'(.+?)是一种(.+)',
            r'(.+?)属于(.+)',
            r'(.+?)包括(.+)',
            r'(.+?)由(.+?)组成',
            r'(.+?)的特点是(.+)'
        ]
        
        for note in atomic_notes:
            content = note.get('content', '')
            note_id = note.get('note_id')
            
            for pattern in definition_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        term, definition = match
                        
                        # 寻找相关的笔记
                        for other_note in atomic_notes:
                            if other_note.get('note_id') == note_id:
                                continue
                            
                            other_content = other_note.get('content', '')
                            
                            # 检查是否包含相关术语
                            if term.strip() in other_content or definition.strip() in other_content:
                                relations.append({
                                    'source_id': note_id,
                                    'target_id': other_note.get('note_id'),
                                    'relation_type': 'definition',
                                    'weight': self.relation_types['definition']['weight'],
                                    'metadata': {
                                        'pattern_type': 'definition_pattern',
                                        'term': term.strip(),
                                        'definition': definition.strip()
                                    }
                                })
        
        return relations
    
    def _extract_causal_pattern_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取因果模式关系"""
        relations = []
        
        causal_patterns = [
            r'因为(.+?)，所以(.+)',
            r'由于(.+?)，导致(.+)',
            r'(.+?)导致了(.+)',
            r'(.+?)的原因是(.+)',
            r'(.+?)造成了(.+)'
        ]
        
        for note in atomic_notes:
            content = note.get('content', '')
            note_id = note.get('note_id')
            
            for pattern in causal_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        cause, effect = match
                        
                        # 寻找包含因果要素的其他笔记
                        for other_note in atomic_notes:
                            if other_note.get('note_id') == note_id:
                                continue
                            
                            other_content = other_note.get('content', '')
                            
                            # 检查因果关系
                            cause_relevance = self._calculate_text_similarity(cause, other_content)
                            effect_relevance = self._calculate_text_similarity(effect, other_content)
                            
                            if cause_relevance > 0.5 or effect_relevance > 0.5:
                                relations.append({
                                    'source_id': note_id,
                                    'target_id': other_note.get('note_id'),
                                    'relation_type': 'causal',
                                    'weight': max(cause_relevance, effect_relevance) * self.relation_types['causal']['weight'],
                                    'metadata': {
                                        'pattern_type': 'causal_pattern',
                                        'cause': cause.strip(),
                                        'effect': effect.strip(),
                                        'relevance': max(cause_relevance, effect_relevance)
                                    }
                                })
        
        return relations
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        # 使用TextUtils的关键词相似度计算方法
        return TextUtils.calculate_similarity_keywords(text1, text2)
    
    def _calculate_reasoning_value(self, relation: Dict[str, Any]) -> float:
        """计算关系的推理价值"""
        relation_type = relation.get('relation_type', 'unknown')
        base_reasoning_value = self.relation_types.get(relation_type, {}).get('reasoning_value', 0.5)
        
        # 基于元数据调整推理价值
        metadata = relation.get('metadata', {})
        confidence = metadata.get('confidence', 0.5)
        
        # 推理价值 = 基础价值 * 置信度 * 权重
        reasoning_value = base_reasoning_value * confidence * relation.get('weight', 0.5)
        
        return min(reasoning_value, 1.0)
    
    # 继承原有的基础方法
    def _extract_reference_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取引用关系（继承原有实现）"""
        relations = []
        note_id_to_note = {note.get('note_id'): note for note in atomic_notes}
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            content = note.get('content', '')
            
            # 查找引用模式
            references = self._find_references_in_text(content, set(note_id_to_note.keys()))
            
            for ref_note_id in references:
                if ref_note_id != note_id and ref_note_id in note_id_to_note:
                    relation = {
                        'source_id': note_id,
                        'target_id': ref_note_id,
                        'relation_type': 'reference',
                        'weight': self.relation_types['reference']['weight'],
                        'metadata': {
                            'reference_context': self._extract_reference_context(content, ref_note_id),
                            'reference_type': 'explicit'
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def _find_references_in_text(self, text: str, available_note_ids: Set[str]) -> List[str]:
        """在文本中查找引用"""
        references = []
        
        # 查找明确的引用模式
        patterns = [
            r'\[([^\]]+)\]',  # [note_id]
            r'@([\w\-_]+)',   # @note_id
            r'见\s*([\w\-_]+)',  # 见 note_id
            r'参考\s*([\w\-_]+)',  # 参考 note_id
            r'如\s*([\w\-_]+)\s*所述',  # 如 note_id 所述
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match in available_note_ids:
                    references.append(match)
        
        return list(set(references))
    
    def _extract_reference_context(self, text: str, ref_note_id: str) -> str:
        """提取引用的上下文"""
        # 查找引用周围的文本
        patterns = [
            rf'(.{{0,50}}\[{re.escape(ref_note_id)}\].{{0,50}})',
            rf'(.{{0,50}}@{re.escape(ref_note_id)}.{{0,50}})',
            rf'(.{{0,50}}{re.escape(ref_note_id)}.{{0,50}})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_entity_coexistence_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取实体共存关系（继承并增强原有实现）"""
        relations = []
        
        # 构建实体到笔记的映射
        entity_to_notes = defaultdict(set)
        note_to_entities = {}
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            entities = note.get('entities', [])
            note_to_entities[note_id] = set(entities)
            
            for entity in entities:
                entity_to_notes[entity].add(note_id)
        
        # 找到共享实体的笔记对
        processed_pairs = set()
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            note_entities = note_to_entities.get(note_id, set())
            
            if not note_entities:
                continue
            
            # 找到共享实体的其他笔记
            related_notes = set()
            for entity in note_entities:
                related_notes.update(entity_to_notes[entity])
            
            related_notes.discard(note_id)
            
            for related_note_id in related_notes:
                # 避免重复处理
                pair = tuple(sorted([note_id, related_note_id]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                # 计算共同实体
                related_entities = note_to_entities.get(related_note_id, set())
                common_entities = note_entities & related_entities
                
                if len(common_entities) >= self.entity_cooccurrence_threshold:
                    # 计算权重（基于共同实体数量和类型）
                    weight = self.relation_types['entity_coexistence']['weight'] * \
                           min(len(common_entities) / 5.0, 1.0)
                    
                    relation = {
                        'source_id': note_id,
                        'target_id': related_note_id,
                        'relation_type': 'entity_coexistence',
                        'weight': weight,
                        'metadata': {
                            'common_entities': list(common_entities),
                            'entity_count': len(common_entities),
                            'jaccard_similarity': len(common_entities) / len(note_entities | related_entities)
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def _extract_context_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取源文档上下文关系（继承原有实现）"""
        relations = []
        
        # 按源文档分组
        doc_to_notes = defaultdict(list)
        for note in atomic_notes:
            source_info = note.get('source_info', {})
            file_path = source_info.get('file_path', '')
            if file_path:
                doc_to_notes[file_path].append(note)
        
        # 为每个文档内的笔记建立上下文关系
        for file_path, notes in doc_to_notes.items():
            if len(notes) < 2:
                continue
            
            # 按位置排序（如果有位置信息）
            notes_with_position = []
            for note in notes:
                source_info = note.get('source_info', {})
                position = source_info.get('chunk_index', 0)
                notes_with_position.append((position, note))
            
            notes_with_position.sort(key=lambda x: x[0])
            sorted_notes = [note for _, note in notes_with_position]
            
            # 建立相邻和近邻关系
            for i, note in enumerate(sorted_notes):
                note_id = note.get('note_id')
                
                # 相邻关系（前后context_window个笔记）
                for j in range(max(0, i - self.context_window), 
                              min(len(sorted_notes), i + self.context_window + 1)):
                    if i == j:
                        continue
                    
                    target_note = sorted_notes[j]
                    target_id = target_note.get('note_id')
                    
                    # 计算距离权重
                    distance = abs(i - j)
                    distance_weight = 1.0 / distance if distance > 0 else 1.0
                    weight = self.relation_types['context_relation']['weight'] * distance_weight
                    
                    relation = {
                        'source_id': note_id,
                        'target_id': target_id,
                        'relation_type': 'context_relation',
                        'weight': weight,
                        'metadata': {
                            'source_document': file_path,
                            'position_distance': distance,
                            'relation_subtype': 'adjacent' if distance == 1 else 'nearby'
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def _extract_topic_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取主题池关系（继承原有实现）"""
        relations = []
        
        # 按主题分组
        topic_to_notes = defaultdict(list)
        for note in atomic_notes:
            cluster_id = note.get('cluster_id')
            topic = note.get('topic', '')
            
            # 使用cluster_id或topic作为分组键
            group_key = cluster_id if cluster_id is not None else topic
            if group_key:
                topic_to_notes[group_key].append(note)
        
        # 为同一主题的笔记建立关系
        for topic, notes in topic_to_notes.items():
            if len(notes) < 2:
                continue
            
            # 计算主题内的关系强度
            topic_weight = self._calculate_topic_cohesion(notes)
            
            # 建立主题内的关系
            for i, note1 in enumerate(notes):
                for j, note2 in enumerate(notes[i + 1:], i + 1):
                    note1_id = note1.get('note_id')
                    note2_id = note2.get('note_id')
                    
                    # 计算笔记间的主题相关性
                    note_similarity = self._calculate_note_topic_similarity(note1, note2)
                    weight = self.relation_types['topic_relation']['weight'] * topic_weight * note_similarity
                    
                    relation = {
                        'source_id': note1_id,
                        'target_id': note2_id,
                        'relation_type': 'topic_relation',
                        'weight': weight,
                        'metadata': {
                            'topic': str(topic),
                            'topic_cohesion': topic_weight,
                            'note_similarity': note_similarity
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def _calculate_topic_cohesion(self, notes: List[Dict[str, Any]]) -> float:
        """计算主题内聚性"""
        if len(notes) < 2:
            return 1.0
        
        # 基于关键词重叠计算内聚性
        all_keywords = []
        for note in notes:
            keywords = note.get('keywords', [])
            all_keywords.extend(keywords)
        
        if not all_keywords:
            return 0.5
        
        # 计算关键词频率
        keyword_counts = Counter(all_keywords)
        total_keywords = len(all_keywords)
        unique_keywords = len(keyword_counts)
        
        # 内聚性 = 重复关键词的比例
        repeated_keywords = sum(1 for count in keyword_counts.values() if count > 1)
        cohesion = repeated_keywords / unique_keywords if unique_keywords > 0 else 0.0
        
        return min(cohesion, 1.0)
    
    def _calculate_note_topic_similarity(self, note1: Dict[str, Any], 
                                       note2: Dict[str, Any]) -> float:
        """计算两个笔记的主题相似性"""
        # 基于关键词相似性
        keywords1 = set(note1.get('keywords', []))
        keywords2 = set(note2.get('keywords', []))
        
        if not keywords1 and not keywords2:
            return 0.5
        
        if not keywords1 or not keywords2:
            return 0.1
        
        # Jaccard相似性
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        return jaccard_sim
    
    def _extract_semantic_similarity_relations(self, atomic_notes: List[Dict[str, Any]], 
                                            embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """提取语义相似性关系（继承并增强原有实现）"""
        relations = []
        
        if embeddings.shape[0] != len(atomic_notes):
            logger.warning("Embeddings count doesn't match notes count")
            return relations
        
        # 计算相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # 提取高相似度的笔记对
        for i in range(len(atomic_notes)):
            for j in range(i + 1, len(atomic_notes)):
                similarity = similarity_matrix[i, j]
                
                if similarity >= self.similarity_threshold:
                    note1 = atomic_notes[i]
                    note2 = atomic_notes[j]
                    
                    note1_id = note1.get('note_id')
                    note2_id = note2.get('note_id')
                    
                    weight = self.relation_types['semantic_similarity']['weight'] * similarity
                    
                    relation = {
                        'source_id': note1_id,
                        'target_id': note2_id,
                        'relation_type': 'semantic_similarity',
                        'weight': weight,
                        'metadata': {
                            'cosine_similarity': float(similarity),
                            'similarity_rank': self._get_similarity_rank(i, j, similarity_matrix)
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算相似度矩阵"""
        # 归一化嵌入
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # 计算余弦相似度
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def _get_similarity_rank(self, i: int, j: int, similarity_matrix: np.ndarray) -> int:
        """获取相似度排名"""
        # 获取第i行的所有相似度值
        similarities = similarity_matrix[i]
        
        # 排序并找到第j个元素的排名
        sorted_indices = np.argsort(similarities)[::-1]
        rank = np.where(sorted_indices == j)[0][0] + 1
        
        return int(rank)
    
    def _filter_and_deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤和去重关系"""
        if not relations:
            return []
        
        # 去重：基于source_id, target_id, relation_type
        seen_relations = set()
        unique_relations = []
        
        for relation in relations:
            source_id = relation.get('source_id')
            target_id = relation.get('target_id')
            relation_type = relation.get('relation_type')
            
            # 创建关系键（双向）
            key1 = (source_id, target_id, relation_type)
            key2 = (target_id, source_id, relation_type)
            
            if key1 not in seen_relations and key2 not in seen_relations:
                seen_relations.add(key1)
                unique_relations.append(relation)
        
        # 按权重排序
        unique_relations.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        # 限制每个笔记的最大关系数
        note_relation_count = defaultdict(int)
        filtered_relations = []
        
        for relation in unique_relations:
            source_id = relation.get('source_id')
            target_id = relation.get('target_id')
            
            if (note_relation_count[source_id] < self.max_relations_per_note and
                note_relation_count[target_id] < self.max_relations_per_note):
                
                filtered_relations.append(relation)
                note_relation_count[source_id] += 1
                note_relation_count[target_id] += 1
        
        return filtered_relations