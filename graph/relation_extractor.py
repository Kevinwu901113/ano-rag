import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
from utils import TextUtils, GPUUtils, BatchProcessor
from config import config

class RelationExtractor:
    """关系提取器，负责从原子笔记中提取各种类型的关系"""
    
    def __init__(self):
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
        
        # 关系类型权重
        self.relation_weights = {
            'reference': config.get('graph.weights.reference', 1.0),
            'entity_coexistence': config.get('graph.weights.entity_coexistence', 0.8),
            'context_relation': config.get('graph.weights.context_relation', 0.6),
            'topic_relation': config.get('graph.weights.topic_relation', 0.7),
            'semantic_similarity': config.get('graph.weights.semantic_similarity', 0.5),
            'personal_relation': config.get('graph.weights.personal_relation', 0.9)
        }
        
        logger.info("RelationExtractor initialized")
    
    def extract_all_relations(self, atomic_notes: List[Dict[str, Any]], 
                             embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """提取所有类型的关系"""
        if not atomic_notes:
            return []
        
        logger.info(f"Extracting relations from {len(atomic_notes)} atomic notes")
        
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
        
        # 去重和过滤
        filtered_relations = self._filter_and_deduplicate_relations(all_relations)
        
        logger.info(f"Extracted {len(filtered_relations)} relations ({len(all_relations)} before filtering)")
        return filtered_relations
    
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
                    weight = self.relation_weights['context_relation'] * distance_weight
                    
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
                    weight = self.relation_weights['topic_relation'] * topic_weight * note_similarity
                    
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
                                'weight': self.relation_weights['personal_relation'],
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
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = embeddings / norms
        
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