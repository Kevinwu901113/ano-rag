import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
from utils import TextUtils, GPUUtils, BatchProcessor
from config import config
from vector_store.embedding_manager import EmbeddingManager

class RelationExtractor:
    """统一的关系提取器，支持基础关系提取和LLM增强的语义关系提取"""
    
    def __init__(self, local_llm=None):
        # LLM支持（可选）
        self.local_llm = local_llm
        self.llm_enabled = local_llm is not None
        
        # 增强配置
        self.enhanced_config = config.get('enhanced_relation_extraction', {})
        self.use_llm_extraction = self.enhanced_config.get('use_llm_extraction', False) and self.llm_enabled
        self.use_fast_model = self.enhanced_config.get('use_fast_model', True)
        self.enable_topic_groups = self.enhanced_config.get('enable_topic_groups', True)
        self.enable_reasoning_paths = self.enhanced_config.get('enable_reasoning_paths', True)
        # 配置参数
        self.similarity_threshold = config.get('graph.similarity_threshold', 0.7)
        self.entity_cooccurrence_threshold = config.get('graph.entity_cooccurrence_threshold', 2)
        self.context_window = config.get('graph.context_window', 3)
        self.max_relations_per_note = config.get('graph.max_relations_per_note', 10)
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=config.get('graph.batch_size', 64),
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # 扩展的关系类型权重和推理价值
        self.relation_types = {
            'reference': {'weight': config.get('graph.weights.reference', 1.0), 'reasoning_value': 0.4},
            'entity_coexistence': {'weight': config.get('graph.weights.entity_coexistence', 0.8), 'reasoning_value': 0.3},
            'context': {'weight': config.get('graph.weights.context_relation', 0.6), 'reasoning_value': 0.5},
            'topic': {'weight': config.get('graph.weights.topic_relation', 0.7), 'reasoning_value': 0.4},
            'semantic_similarity': {'weight': config.get('graph.weights.semantic_similarity', 0.5), 'reasoning_value': 0.6},
            'personal': {'weight': config.get('graph.weights.personal_relation', 0.9), 'reasoning_value': 0.3},
            # 增强关系类型
            'causal': {'weight': 0.9, 'reasoning_value': 1.0},
            'temporal': {'weight': 0.8, 'reasoning_value': 0.8},
            'definition': {'weight': 0.7, 'reasoning_value': 0.7},
            'comparison': {'weight': 0.6, 'reasoning_value': 0.6},
            'elaboration': {'weight': 0.5, 'reasoning_value': 0.5},
            'contradiction': {'weight': 0.8, 'reasoning_value': 0.9},
            # 新增轻量级关系类型
            'succession': {'weight': 0.85, 'reasoning_value': 0.9},  # 继任关系
            'acquisition': {'weight': 0.9, 'reasoning_value': 0.95}, # 收购关系
            'ownership': {'weight': 0.8, 'reasoning_value': 0.8},    # 归属关系
            'subsidiary': {'weight': 0.75, 'reasoning_value': 0.7},  # 子公司关系
            'partnership': {'weight': 0.7, 'reasoning_value': 0.6},  # 合作关系
            'merger': {'weight': 0.9, 'reasoning_value': 0.95}       # 合并关系
        }
        
        # 向后兼容的权重字典
        self.relation_weights = {k: v['weight'] for k, v in self.relation_types.items()}
        
        logger.info("RelationExtractor initialized")
    
    def extract_all_relations(self, atomic_notes: List[Dict[str, Any]], 
                             embeddings: Optional[np.ndarray] = None,
                             topic_groups: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
        """提取所有类型的关系（基础+增强）"""
        if not atomic_notes:
            return []
        
        logger.info(f"Extracting relations from {len(atomic_notes)} atomic notes")
        
        all_relations = []
        
        # 1. 基础关系提取
        basic_relations = self._extract_basic_relations(atomic_notes, embeddings)
        all_relations.extend(basic_relations)
        
        # 2. 主题组关系（如果启用）
        if self.enable_topic_groups and topic_groups:
            logger.info("Extracting topic group relations")
            topic_group_relations = self._extract_topic_group_relations(atomic_notes, topic_groups)
            all_relations.extend(topic_group_relations)
        
        # 3. LLM增强的语义关系（如果启用）
        if self.use_llm_extraction:
            logger.info("Extracting LLM-enhanced semantic relations")
            llm_relations = self._extract_llm_semantic_relations(atomic_notes)
            all_relations.extend(llm_relations)
        
        # 4. 推理路径关系（如果启用）
        # if self.enable_reasoning_paths:
        #     logger.info("Extracting reasoning path relations")
        #     reasoning_relations = self._extract_reasoning_path_relations(all_relations, atomic_notes)
        #     all_relations.extend(reasoning_relations)
        
        # 过滤、去重和计算推理价值
        filtered_relations = self._filter_and_deduplicate_relations(all_relations)
        enhanced_relations = self._calculate_reasoning_values(filtered_relations)
        
        logger.info(f"Extracted {len(enhanced_relations)} total relations")
        return enhanced_relations
    
    def _extract_basic_relations(self, atomic_notes: List[Dict[str, Any]], 
                               embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """提取基础关系"""
        all_relations = []
        
        # 1. 引用关系
        logger.info("Extracting reference relations")
        reference_relations = self.extract_reference_relations(atomic_notes)
        all_relations.extend(reference_relations)
        
        # 2. 实体共存关系
        logger.info("Extracting entity coexistence relations")
        entity_relations = self.extract_entity_coexistence_relations(atomic_notes)
        all_relations.extend(entity_relations)
        
        # 3. 源文档上下文关系
        logger.info("Extracting context relations")
        context_relations = self.extract_context_relations(atomic_notes)
        all_relations.extend(context_relations)
        
        # 4. 主题池关系
        logger.info("Extracting topic relations")
        topic_relations = self.extract_topic_relations(atomic_notes)
        all_relations.extend(topic_relations)
        
        # 5. 语义相似性关系（如果提供了嵌入）
        if embeddings is not None:
            logger.info("Extracting semantic similarity relations")
            similarity_relations = self.extract_semantic_similarity_relations(
                atomic_notes, embeddings
            )
            all_relations.extend(similarity_relations)

        # 6. 个人关系
        logger.info("Extracting personal relations")
        personal_relations = self.extract_personal_relations(atomic_notes)
        all_relations.extend(personal_relations)
        
        # 7. 轻量级业务关系
        logger.info("Extracting lightweight business relations")
        business_relations = self.extract_lightweight_business_relations(atomic_notes)
        all_relations.extend(business_relations)
        
        return all_relations
    
    def _extract_topic_group_relations(self, atomic_notes: List[Dict[str, Any]], 
                                     topic_groups: List[List[str]]) -> List[Dict[str, Any]]:
        """提取主题组内的关系"""
        relations = []
        note_id_to_note = {note['note_id']: note for note in atomic_notes}
        
        for group in topic_groups:
            if len(group) < 2:
                continue
            
            # 在同一主题组内的笔记之间建立关系
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    note1_id, note2_id = group[i], group[j]
                    
                    if note1_id in note_id_to_note and note2_id in note_id_to_note:
                        note1 = note_id_to_note[note1_id]
                        note2 = note_id_to_note[note2_id]
                        
                        # 计算主题相似度
                        topic_similarity = self._calculate_note_topic_similarity(note1, note2)
                        
                        if topic_similarity > 0.3:  # 主题相似度阈值
                            relations.append({
                                'source_id': note1_id,
                                'target_id': note2_id,
                                'relation_type': 'topic_group',
                                'weight': topic_similarity * self.relation_types['topic']['weight'],
                                'metadata': {
                                    'topic_similarity': topic_similarity,
                                    'group_size': len(group)
                                }
                            })
        
        logger.info(f"Extracted {len(relations)} topic group relations")
        return relations
    
    def _extract_llm_semantic_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用LLM提取语义关系"""
        if not self.llm_enabled:
            return []
        
        relations = []
        
        # 选择模型
        model_name = 'fast_model' if self.use_fast_model else 'default_model'
        
        # 批量处理笔记对
        note_pairs = []
        for i in range(len(atomic_notes)):
            for j in range(i + 1, min(i + 10, len(atomic_notes))):  # 限制每个笔记的比较数量
                note_pairs.append((atomic_notes[i], atomic_notes[j]))
        
        # 分批处理
        batch_size = 5
        for i in range(0, len(note_pairs), batch_size):
            batch = note_pairs[i:i + batch_size]
            batch_relations = self._process_llm_relation_batch(batch, model_name)
            relations.extend(batch_relations)
        
        logger.info(f"Extracted {len(relations)} LLM semantic relations")
        return relations
    
    def _process_llm_relation_batch(self, note_pairs: List[Tuple[Dict, Dict]], 
                                  model_name: str) -> List[Dict[str, Any]]:
        """处理一批笔记对的LLM关系提取"""
        relations = []
        
        for note1, note2 in note_pairs:
            try:
                # 构建提示
                prompt = self._build_relation_extraction_prompt(note1, note2)
                
                # 调用LLM
                response = self.local_llm.generate(
                    prompt=prompt,
                    model_name=model_name,
                    max_tokens=200,
                    temperature=0.1
                )
                
                # 解析响应
                extracted_relations = self._parse_llm_relation_response(response, note1['note_id'], note2['note_id'])
                relations.extend(extracted_relations)
                
            except Exception as e:
                logger.warning(f"LLM relation extraction failed for {note1['note_id']}-{note2['note_id']}: {e}")
                continue
        
        return relations
    
    def _build_relation_extraction_prompt(self, note1: Dict, note2: Dict) -> str:
        """构建关系提取的提示"""
        prompt = f"""分析以下两个笔记之间的语义关系：

笔记1: {note1.get('title', '')}
内容: {note1.get('content', '')[:200]}...

笔记2: {note2.get('title', '')}
内容: {note2.get('content', '')[:200]}...

请识别它们之间的关系类型，可能的关系包括：
- causal: 因果关系
- temporal: 时间关系
- definition: 定义关系
- comparison: 比较关系
- elaboration: 详细说明关系
- contradiction: 矛盾关系

如果存在关系，请回答格式：关系类型|置信度(0-1)|简短说明
如果不存在明显关系，请回答：none"""
        
        return prompt
    
    def _parse_llm_relation_response(self, response: str, note1_id: str, note2_id: str) -> List[Dict[str, Any]]:
        """解析LLM关系提取响应"""
        relations = []
        
        if not response or response.strip().lower() == 'none':
            return relations
        
        try:
            parts = response.strip().split('|')
            if len(parts) >= 2:
                relation_type = parts[0].strip()
                confidence = float(parts[1].strip())
                explanation = parts[2].strip() if len(parts) > 2 else ''
                
                if relation_type in self.relation_types and confidence > 0.5:
                    relations.append({
                        'source_id': note1_id,
                        'target_id': note2_id,
                        'relation_type': relation_type,
                        'weight': confidence * self.relation_types[relation_type]['weight'],
                        'metadata': {
                            'confidence': confidence,
                            'explanation': explanation,
                            'extraction_method': 'llm'
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {response}, error: {e}")
        
        return relations
    
    def _extract_reasoning_path_relations(self, existing_relations: List[Dict[str, Any]], 
                                        atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于现有关系提取推理路径关系"""
        relations = []
        
        # 构建关系图
        relation_graph = defaultdict(list)
        for rel in existing_relations:
            relation_graph[rel['source_id']].append(rel)
        
        # 寻找间接关系（2跳路径）
        for note in atomic_notes:
            note_id = note['note_id']
            
            # 找到从该笔记出发的直接关系
            direct_relations = relation_graph.get(note_id, [])
            
            for direct_rel in direct_relations:
                target_id = direct_rel['target_id']
                
                # 找到从目标笔记出发的关系
                indirect_relations = relation_graph.get(target_id, [])
                
                for indirect_rel in indirect_relations:
                    final_target = indirect_rel['target_id']
                    
                    # 避免自环和已存在的直接关系
                    if final_target != note_id and not self._relation_exists(existing_relations, note_id, final_target):
                        # 计算推理路径强度
                        path_strength = direct_rel['weight'] * indirect_rel['weight'] * 0.5  # 衰减因子
                        
                        if path_strength > 0.1:  # 最小强度阈值
                            relations.append({
                                'source_id': note_id,
                                'target_id': final_target,
                                'relation_type': 'reasoning_path',
                                'weight': path_strength,
                                'metadata': {
                                    'path': [note_id, target_id, final_target],
                                    'intermediate_relations': [
                                        direct_rel['relation_type'],
                                        indirect_rel['relation_type']
                                    ],
                                    'path_strength': path_strength
                                }
                            })
        
        logger.info(f"Extracted {len(relations)} reasoning path relations")
        return relations
    
    def _relation_exists(self, relations: List[Dict[str, Any]], source_id: str, target_id: str) -> bool:
        """检查关系是否已存在"""
        for rel in relations:
            if ((rel['source_id'] == source_id and rel['target_id'] == target_id) or
                (rel['source_id'] == target_id and rel['target_id'] == source_id)):
                return True
        return False
    
    def _calculate_reasoning_values(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为关系计算推理价值"""
        for relation in relations:
            relation_type = relation['relation_type']
            if relation_type in self.relation_types:
                reasoning_value = self.relation_types[relation_type]['reasoning_value']
                relation['reasoning_value'] = reasoning_value
            else:
                relation['reasoning_value'] = 0.5  # 默认值
        
        return relations
    
    # 使用下面已有的_calculate_note_topic_similarity方法
    
    def _group_notes_by_topic(self, atomic_notes: List[Dict[str, Any]]) -> List[List[str]]:
        """根据主题对笔记进行分组"""
        topic_groups = []
        
        # 简单的基于关键词的分组
        topic_to_notes = defaultdict(list)
        
        for note in atomic_notes:
            topics = note.get('topics', [])
            if topics:
                # 使用第一个主题作为分组依据
                main_topic = topics[0]
                topic_to_notes[main_topic].append(note['note_id'])
        
        # 只保留包含多个笔记的组
        for topic, note_ids in topic_to_notes.items():
            if len(note_ids) > 1:
                topic_groups.append(note_ids)
        
        return topic_groups
    
    # 原有的基础关系提取方法保持不变
    
    def extract_reference_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取引用关系"""
        relations = []
        note_id_to_note = {note.get('note_id'): note for note in atomic_notes}
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            content = note.get('content', '')
            
            # 查找引用模式
            references = self._find_references_in_text(content, note_id_to_note.keys())
            
            for ref_note_id in references:
                if ref_note_id != note_id and ref_note_id in note_id_to_note:
                    relation = {
                        'source_id': note_id,
                        'target_id': ref_note_id,
                        'relation_type': 'reference',
                        'weight': self.relation_weights['reference'],
                        'metadata': {
                            'reference_context': self._extract_reference_context(content, ref_note_id),
                            'reference_type': 'explicit'
                        }
                    }
                    relations.append(relation)
        
        return relations
    
    def extract_entity_coexistence_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取实体共存关系"""
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
                
                # Allow pairs that share at least one entity
                if len(common_entities) >= max(self.entity_cooccurrence_threshold, 1):
                    # 计算权重（基于共同实体数量）
                    weight = self.relation_weights['entity_coexistence'] * \
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
    
    def extract_context_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取源文档上下文关系"""
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
                    weight = self.relation_weights['context'] * distance_weight
                    
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
    
    def extract_topic_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取主题池关系"""
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
                    weight = self.relation_weights['topic'] * topic_weight * note_similarity
                    
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
    
    def extract_semantic_similarity_relations(self, atomic_notes: List[Dict[str, Any]], 
                                            embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """提取语义相似性关系"""
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
                    
                    weight = self.relation_weights['semantic_similarity'] * similarity
                    
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

    def extract_personal_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解析婚姻等个人关系"""
        relations = []

        # 构建实体到笔记的映射
        entity_to_notes = defaultdict(list)
        for note in atomic_notes:
            note_id = note.get('note_id')
            for entity in note.get('entities', []):
                entity_to_notes[entity].append(note_id)

        patterns = [
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*) is married to ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'s wife ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'s husband ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'s spouse ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        ]

        for note in atomic_notes:
            content = note.get('content', '')
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    name1 = match.group(1).strip()
                    name2 = match.group(2).strip()
                    snippet = match.group(0).strip()
                    ids1 = entity_to_notes.get(name1, [])
                    ids2 = entity_to_notes.get(name2, [])
                    for id1 in ids1:
                        for id2 in ids2:
                            if id1 == id2:
                                continue
                            relations.append({
                                'source_id': id1,
                                'target_id': id2,
                                'relation_type': 'personal_relation',
                                'weight': self.relation_weights['personal'],
                                'metadata': {
                                    'relation_subtype': 'spouse',
                                    'confidence': 0.9,
                                    'snippet': snippet
                                }
                            })

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
        
        # 基于实体相似性
        entities1 = set(note1.get('entities', []))
        entities2 = set(note2.get('entities', []))
        
        entity_intersection = len(entities1 & entities2)
        entity_union = len(entities1 | entities2)
        entity_sim = entity_intersection / entity_union if entity_union > 0 else 0.0
        
        # 综合相似性
        combined_sim = 0.7 * jaccard_sim + 0.3 * entity_sim
        return min(combined_sim, 1.0)
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算相似度矩阵"""
        # 归一化嵌入
        normalized_embeddings = EmbeddingManager.normalize_embeddings(embeddings)
        
        # 计算余弦相似度
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # 将对角线设为0（避免自相似）
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _get_similarity_rank(self, source_index: int, target_index: int,
                             similarity_matrix: np.ndarray) -> int:
        """获取目标笔记在源笔记相似度排序中的排名"""
        similarities = similarity_matrix[source_index]
        # 排序并找到排名
        sorted_indices = np.argsort(similarities)[::-1]
        rank = np.where(sorted_indices == target_index)[0][0] + 1
        return int(rank)
    
    def _filter_and_deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化的关系过滤和去重"""
        if not relations:
            return []
        
        # 按source_id, target_id, relation_type去重，保留权重最高的
        unique_relations = {}
        for relation in relations:
            source_id = relation.get('source_id')
            target_id = relation.get('target_id')
            relation_type = relation.get('relation_type')
            
            # 跳过无效的关系
            if source_id is None or target_id is None or relation_type is None:
                continue
                
            # 创建标准化的关系键（确保方向一致性）
            if relation_type in ['entity_coexistence', 'semantic_similarity', 'topic_relation', 'personal_relation']:
                # 这些关系是无向的
                relation_key = (min(source_id, target_id), max(source_id, target_id), relation_type)
            else:
                # 有向关系
                relation_key = (source_id, target_id, relation_type)
            
            if relation_key not in unique_relations or relation.get('weight', 0) > unique_relations[relation_key].get('weight', 0):
                unique_relations[relation_key] = relation
        
        filtered_relations = list(unique_relations.values())
        
        # 基于关系类型的重要性过滤
        filtered_relations = self._filter_relations_by_importance(filtered_relations)
        
        # 按权重排序
        filtered_relations.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        # 动态限制每个笔记的关系数量
        final_relations = self._apply_dynamic_relation_limits(filtered_relations)
        
        logger.info(f"Filtered relations: {len(final_relations)} from {len(relations)}")
        return final_relations
    
    def _filter_relations_by_importance(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于重要性的关系过滤"""
        # 按关系类型分组
        relation_groups = defaultdict(list)
        for rel in relations:
            rel_type = rel.get('relation_type')
            relation_groups[rel_type].append(rel)
        
        filtered_relations = []
        
        # 为不同类型的关系设置不同的保留策略
        type_limits = {
            'reference': 100,  # 引用关系最重要
            'entity_coexistence': 60,
            'context_relation': 40,
            'topic_relation': 30,
            'semantic_similarity': 50,
            'personal_relation': 80
        }
        
        for rel_type, rels in relation_groups.items():
            limit = type_limits.get(rel_type, 25)
            # 按权重排序并取前N个
            sorted_rels = sorted(rels, key=lambda x: x.get('weight', 0), reverse=True)
            filtered_relations.extend(sorted_rels[:limit])
        
        return filtered_relations
    
    def _apply_dynamic_relation_limits(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用动态关系数量限制"""
        note_relation_counts = defaultdict(int)
        note_importance = defaultdict(float)
        
        # 计算笔记重要性（基于关系数量和权重）
        for relation in relations:
            source_id = relation.get('source_id')
            target_id = relation.get('target_id')
            weight = relation.get('weight', 0)
            
            note_importance[source_id] += weight
            note_importance[target_id] += weight
        
        # 为重要笔记分配更多关系配额
        final_relations = []
        for relation in relations:
            source_id = relation.get('source_id')
            target_id = relation.get('target_id')
            
            # 基于重要性调整限制
            source_limit = min(self.max_relations_per_note * 2, 
                             self.max_relations_per_note + int(note_importance[source_id] / 10))
            target_limit = min(self.max_relations_per_note * 2, 
                             self.max_relations_per_note + int(note_importance[target_id] / 10))
            
            if (note_relation_counts[source_id] < source_limit and 
                note_relation_counts[target_id] < target_limit):
                final_relations.append(relation)
                note_relation_counts[source_id] += 1
                note_relation_counts[target_id] += 1
        
        return final_relations
    
    def get_relation_statistics(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取关系统计信息"""
        if not relations:
            return {}
        
        # 按类型统计
        type_counts = Counter(r.get('relation_type') for r in relations)
        
        # 权重统计
        weights = [r.get('weight', 0) for r in relations]
        
        # 节点度数统计
        node_degrees = defaultdict(int)
        for relation in relations:
            node_degrees[relation.get('source_id')] += 1
            node_degrees[relation.get('target_id')] += 1
        
        stats = {
            'total_relations': len(relations),
            'relation_types': dict(type_counts),
            'weight_stats': {
                'mean': np.mean(weights) if weights else 0,
                'std': np.std(weights) if weights else 0,
                'min': np.min(weights) if weights else 0,
                'max': np.max(weights) if weights else 0
            },
            'node_degree_stats': {
                'mean': np.mean(list(node_degrees.values())) if node_degrees else 0,
                'max': max(node_degrees.values()) if node_degrees else 0,
                'min': min(node_degrees.values()) if node_degrees else 0
            },
            'unique_nodes': len(node_degrees)
        }
        
        return stats
    
    def extract_relations_batch(self, atomic_notes_batches: List[List[Dict[str, Any]]], 
                               embeddings_batches: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """批量提取关系"""
        all_relations = []
        
        for i, notes_batch in enumerate(atomic_notes_batches):
            embeddings_batch = embeddings_batches[i] if embeddings_batches else None
            
            logger.info(f"Processing batch {i + 1}/{len(atomic_notes_batches)}")
            batch_relations = self.extract_all_relations(notes_batch, embeddings_batch)
            all_relations.extend(batch_relations)
        
        # 最终过滤和去重
        return self._filter_and_deduplicate_relations(all_relations)
    
    def extract_lightweight_business_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取轻量级业务关系（继任、收购、归属等）"""
        # 检查是否启用轻量级关系抽取
        lightweight_config = config.get('relation_extractor.lightweight_relations', {})
        if not lightweight_config.get('enabled', True):
            return []
        
        # 获取启用的关系类型
        enabled_types = lightweight_config.get('types', 
                                 ['succession', 'acquisition', 'ownership', 'subsidiary', 'partnership', 'merger'])
        
        relations = []
        
        # 定义关系模式和关键词
        relation_patterns = {
            'succession': {
                'keywords': ['继任', '接任', '接替', '继承', '接班', '替代', '取代', 'succeed', 'replace', 'take over'],
                'patterns': [r'(\w+)\s*继任\s*(\w+)', r'(\w+)\s*接替\s*(\w+)', r'(\w+)\s*succeed\s*(\w+)']
            },
            'acquisition': {
                'keywords': ['收购', '并购', '兼并', '购买', '买下', 'acquire', 'purchase', 'buy out', 'takeover'],
                'patterns': [r'(\w+)\s*收购\s*(\w+)', r'(\w+)\s*并购\s*(\w+)', r'(\w+)\s*acquire[sd]?\s*(\w+)']
            },
            'ownership': {
                'keywords': ['拥有', '持有', '控制', '所有', '归属', 'own', 'control', 'possess', 'belong to'],
                'patterns': [r'(\w+)\s*拥有\s*(\w+)', r'(\w+)\s*控制\s*(\w+)', r'(\w+)\s*own[s]?\s*(\w+)']
            },
            'subsidiary': {
                'keywords': ['子公司', '分公司', '附属', '下属', 'subsidiary', 'affiliate', 'branch'],
                'patterns': [r'(\w+)\s*的?\s*子公司\s*(\w+)', r'(\w+)\s*subsidiary\s*(\w+)']
            },
            'partnership': {
                'keywords': ['合作', '合伙', '联盟', '伙伴', 'partner', 'collaborate', 'alliance', 'cooperation'],
                'patterns': [r'(\w+)\s*与\s*(\w+)\s*合作', r'(\w+)\s*partner[s]?\s*with\s*(\w+)']
            },
            'merger': {
                'keywords': ['合并', '融合', '整合', 'merge', 'consolidate', 'integrate'],
                'patterns': [r'(\w+)\s*与\s*(\w+)\s*合并', r'(\w+)\s*merge[sd]?\s*with\s*(\w+)']
            }
        }
        
        for i, note1 in enumerate(atomic_notes):
            note1_id = note1.get('note_id')
            if not note1_id:
                continue
                
            text = note1.get('content', '') + ' ' + note1.get('original_text', '')
            entities = note1.get('entities', [])
            
            # 检查每种关系类型（仅处理启用的类型）
            for relation_type, pattern_config in relation_patterns.items():
                if relation_type not in enabled_types:
                    continue
                # 关键词匹配
                if any(keyword in text for keyword in pattern_config['keywords']):
                    # 模式匹配
                    for pattern in pattern_config['patterns']:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if len(match.groups()) >= 2:
                                entity1, entity2 = match.groups()[:2]
                                if entity1 and entity2 and entity1 != entity2:
                                    # 查找包含这些实体的其他笔记
                                    for j, note2 in enumerate(atomic_notes):
                                        if i == j:
                                            continue
                                        note2_id = note2.get('note_id')
                                        if not note2_id:
                                            continue
                                        
                                        note2_entities = note2.get('entities', [])
                                        if (entity1 in note2_entities or entity2 in note2_entities or
                                            entity1 in note2.get('content', '') or entity2 in note2.get('content', '')):
                                            
                                            weight = self.relation_types[relation_type]['weight']
                                            relations.append({
                                                'source_id': note1_id,
                                                'target_id': note2_id,
                                                'relation_type': relation_type,
                                                'weight': weight,
                                                'metadata': {
                                                    'entity1': entity1,
                                                    'entity2': entity2,
                                                    'pattern_match': match.group(),
                                                    'extraction_method': 'lightweight_business_pattern'
                                                }
                                            })
                
                # 实体对关系检查
                if len(entities) >= 2:
                    for k in range(len(entities)):
                        for l in range(k + 1, len(entities)):
                            entity1, entity2 = entities[k], entities[l]
                            
                            # 检查是否存在关系关键词
                            if any(keyword in text for keyword in pattern_config['keywords']):
                                # 查找相关的其他笔记
                                for j, note2 in enumerate(atomic_notes):
                                    if i == j:
                                        continue
                                    note2_id = note2.get('note_id')
                                    if not note2_id:
                                        continue
                                    
                                    note2_entities = note2.get('entities', [])
                                    if entity1 in note2_entities or entity2 in note2_entities:
                                        weight = self.relation_types[relation_type]['weight'] * 0.8  # 稍微降低权重
                                        relations.append({
                                            'source_id': note1_id,
                                            'target_id': note2_id,
                                            'relation_type': relation_type,
                                            'weight': weight,
                                            'metadata': {
                                                'entity1': entity1,
                                                'entity2': entity2,
                                                'extraction_method': 'lightweight_business_entity_pair'
                                            }
                                        })
        
        logger.info(f"Extracted {len(relations)} lightweight business relations")
        return relations