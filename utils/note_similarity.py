from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import json
from collections import defaultdict
import threading
import time

class NoteSimilarityCalculator:
    """基于嵌入的笔记相似度计算器，用于构建笔记间的关联关系"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 模型配置
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)
        self.max_related_notes = self.config.get('max_related_notes', 5)
        self.batch_size = self.config.get('batch_size', 32)
        
        # 相似度计算配置
        self.use_content = self.config.get('use_content', True)
        self.use_summary = self.config.get('use_summary', True)
        self.content_weight = self.config.get('content_weight', 0.6)
        self.summary_weight = self.config.get('summary_weight', 0.4)
        
        # 实体相似度配置
        self.entity_similarity_weight = self.config.get('entity_similarity_weight', 0.3)
        self.min_shared_entities = self.config.get('min_shared_entities', 1)
        
        # 聚类过滤配置
        self.exclude_same_cluster = self.config.get('exclude_same_cluster', True)
        self.cluster_similarity_bonus = self.config.get('cluster_similarity_bonus', 0.1)
        
        # 初始化模型
        self.model = None
        self._model_lock = threading.Lock()
        
        # 缓存
        self.embedding_cache = {}
        self.similarity_cache = {}
    
    def _load_model(self):
        """延迟加载sentence transformer模型"""
        if self.model is None:
            with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading sentence transformer model: {self.model_name}")
                    try:
                        self.model = SentenceTransformer(self.model_name)
                        logger.info("Model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load model {self.model_name}: {e}")
                        # 回退到更简单的模型
                        try:
                            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                            logger.info("Loaded fallback model: paraphrase-MiniLM-L3-v2")
                        except Exception as e2:
                            logger.error(f"Failed to load fallback model: {e2}")
                            raise e2
    
    def compute_note_embeddings(self, notes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """计算笔记的嵌入向量"""
        self._load_model()
        
        embeddings = {}
        texts_to_encode = []
        note_ids = []
        
        logger.info(f"Computing embeddings for {len(notes)} notes")
        
        for note in notes:
            note_id = note.get('id', str(hash(note.get('content', ''))))
            
            # 检查缓存
            cache_key = self._get_cache_key(note)
            if cache_key in self.embedding_cache:
                embeddings[note_id] = self.embedding_cache[cache_key]
                continue
            
            # 构建用于编码的文本
            text = self._prepare_text_for_encoding(note)
            if text:
                texts_to_encode.append(text)
                note_ids.append(note_id)
        
        # 批量编码
        if texts_to_encode:
            try:
                batch_embeddings = self.model.encode(
                    texts_to_encode, 
                    batch_size=self.batch_size,
                    show_progress_bar=True
                )
                
                # 存储结果和缓存
                for i, note_id in enumerate(note_ids):
                    embedding = batch_embeddings[i]
                    embeddings[note_id] = embedding
                    
                    # 更新缓存
                    note = next(n for n in notes if n.get('id', str(hash(n.get('content', '')))) == note_id)
                    cache_key = self._get_cache_key(note)
                    self.embedding_cache[cache_key] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to compute embeddings: {e}")
                return {}
        
        logger.info(f"Computed embeddings for {len(embeddings)} notes")
        return embeddings
    
    def find_related_notes(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为每个笔记找到相关的笔记"""
        logger.info(f"Finding related notes for {len(notes)} notes")
        
        # 计算嵌入
        embeddings = self.compute_note_embeddings(notes)
        
        if not embeddings:
            logger.warning("No embeddings computed, returning original notes")
            return notes
        
        # 构建笔记ID到索引的映射
        note_id_to_index = {}
        embedding_matrix = []
        valid_notes = []
        
        for i, note in enumerate(notes):
            note_id = note.get('id', str(hash(note.get('content', ''))))
            if note_id in embeddings:
                note_id_to_index[note_id] = len(valid_notes)
                embedding_matrix.append(embeddings[note_id])
                valid_notes.append(note)
        
        if len(embedding_matrix) < 2:
            logger.warning("Not enough valid embeddings for similarity computation")
            return notes
        
        # 计算相似度矩阵
        embedding_matrix = np.array(embedding_matrix)
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # 为每个笔记找到相关笔记
        enhanced_notes = []
        
        for note in notes:
            note_id = note.get('id', str(hash(note.get('content', ''))))
            enhanced_note = note.copy()
            
            if note_id in note_id_to_index:
                related_notes = self._find_related_for_note(
                    note, valid_notes, similarity_matrix, note_id_to_index[note_id]
                )
                enhanced_note['related_notes'] = related_notes
            else:
                enhanced_note['related_notes'] = []
            
            enhanced_notes.append(enhanced_note)
        
        # 统计信息
        total_relations = sum(len(note.get('related_notes', [])) for note in enhanced_notes)
        logger.info(f"Found {total_relations} note relationships")
        
        return enhanced_notes
    
    def _find_related_for_note(self, target_note: Dict[str, Any], 
                              all_notes: List[Dict[str, Any]], 
                              similarity_matrix: np.ndarray, 
                              target_index: int) -> List[Dict[str, Any]]:
        """为特定笔记找到相关笔记"""
        target_cluster = target_note.get('cluster_id', -1)
        related_notes = []
        
        # 获取相似度分数
        similarities = similarity_matrix[target_index]
        
        # 创建候选列表
        candidates = []
        for i, note in enumerate(all_notes):
            if i == target_index:  # 跳过自己
                continue
            
            # 检查是否在同一聚类（如果配置为排除）
            if self.exclude_same_cluster and note.get('cluster_id', -1) == target_cluster and target_cluster != -1:
                continue
            
            # 计算综合相似度
            base_similarity = similarities[i]
            
            # 实体相似度加成
            entity_similarity = self._calculate_entity_similarity(target_note, note)
            
            # 聚类加成（如果不在同一聚类但相似度高）
            cluster_bonus = 0
            if (not self.exclude_same_cluster and 
                note.get('cluster_id', -1) != target_cluster and 
                base_similarity > self.similarity_threshold - 0.1):
                cluster_bonus = self.cluster_similarity_bonus
            
            # 综合相似度
            combined_similarity = (
                base_similarity * (1 - self.entity_similarity_weight) +
                entity_similarity * self.entity_similarity_weight +
                cluster_bonus
            )
            
            if combined_similarity > self.similarity_threshold:
                candidates.append({
                    'note': note,
                    'similarity': combined_similarity,
                    'base_similarity': base_similarity,
                    'entity_similarity': entity_similarity
                })
        
        # 按相似度排序并选择前N个
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        for candidate in candidates[:self.max_related_notes]:
            note = candidate['note']
            related_note = {
                'id': note.get('id', str(hash(note.get('content', '')))),
                'title': note.get('summary', note.get('content', '')[:50] + '...'),
                'similarity': round(candidate['similarity'], 3),
                'base_similarity': round(candidate['base_similarity'], 3),
                'entity_similarity': round(candidate['entity_similarity'], 3),
                'cluster_id': note.get('cluster_id', -1)
            }
            related_notes.append(related_note)
        
        return related_notes
    
    def _calculate_entity_similarity(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> float:
        """计算两个笔记的实体相似度"""
        entities1 = set(note1.get('entities', []))
        entities2 = set(note2.get('entities', []))
        
        if not entities1 and not entities2:
            return 0.5  # 都没有实体，给中性分数
        
        if not entities1 or not entities2:
            return 0.0  # 一个有实体一个没有
        
        # 计算Jaccard相似度
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        
        # 如果共享实体数量达到最小要求，给予奖励
        if len(intersection) >= self.min_shared_entities:
            jaccard_similarity += 0.1
        
        return min(jaccard_similarity, 1.0)
    
    def _prepare_text_for_encoding(self, note: Dict[str, Any]) -> str:
        """准备用于编码的文本"""
        texts = []
        
        if self.use_content and note.get('content'):
            content = note['content'].strip()
            if content:
                texts.append(content)
        
        if self.use_summary and note.get('summary'):
            summary = note['summary'].strip()
            if summary and summary != note.get('content', '').strip():
                texts.append(summary)
        
        # 如果都没有，使用关键词
        if not texts and note.get('keywords'):
            keywords = note['keywords']
            if isinstance(keywords, list):
                texts.append(' '.join(keywords))
            elif isinstance(keywords, str):
                texts.append(keywords)
        
        return ' '.join(texts) if texts else ''
    
    def _get_cache_key(self, note: Dict[str, Any]) -> str:
        """生成缓存键"""
        content = note.get('content', '')
        summary = note.get('summary', '')
        return f"{hash(content)}_{hash(summary)}"
    
    def compute_similarity_statistics(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算相似度统计信息"""
        total_notes = len(notes)
        notes_with_relations = sum(1 for note in notes if note.get('related_notes'))
        total_relations = sum(len(note.get('related_notes', [])) for note in notes)
        
        # 相似度分布
        similarities = []
        for note in notes:
            for related in note.get('related_notes', []):
                similarities.append(related.get('similarity', 0))
        
        # 聚类间关系统计
        cross_cluster_relations = 0
        same_cluster_relations = 0
        
        for note in notes:
            note_cluster = note.get('cluster_id', -1)
            for related in note.get('related_notes', []):
                related_cluster = related.get('cluster_id', -1)
                if note_cluster != related_cluster:
                    cross_cluster_relations += 1
                else:
                    same_cluster_relations += 1
        
        stats = {
            'total_notes': total_notes,
            'notes_with_relations': notes_with_relations,
            'total_relations': total_relations,
            'avg_relations_per_note': total_relations / max(total_notes, 1),
            'coverage_rate': notes_with_relations / max(total_notes, 1),
            'cross_cluster_relations': cross_cluster_relations,
            'same_cluster_relations': same_cluster_relations,
            'similarity_stats': {
                'mean': np.mean(similarities) if similarities else 0,
                'std': np.std(similarities) if similarities else 0,
                'min': np.min(similarities) if similarities else 0,
                'max': np.max(similarities) if similarities else 0
            }
        }
        
        return stats
    
    def export_similarity_graph(self, notes: List[Dict[str, Any]], 
                               output_path: str, format: str = 'json') -> None:
        """导出相似度图"""
        if format == 'json':
            graph_data = {
                'nodes': [],
                'edges': []
            }
            
            # 添加节点
            for note in notes:
                node = {
                    'id': note.get('id', str(hash(note.get('content', '')))),
                    'title': note.get('summary', note.get('content', '')[:50] + '...'),
                    'cluster_id': note.get('cluster_id', -1),
                    'importance_score': note.get('importance_score', 0),
                    'entities': note.get('entities', []),
                    'is_noise': note.get('is_noise', False)
                }
                graph_data['nodes'].append(node)
            
            # 添加边
            for note in notes:
                source_id = note.get('id', str(hash(note.get('content', ''))))
                for related in note.get('related_notes', []):
                    edge = {
                        'source': source_id,
                        'target': related['id'],
                        'similarity': related['similarity'],
                        'base_similarity': related.get('base_similarity', 0),
                        'entity_similarity': related.get('entity_similarity', 0)
                    }
                    graph_data['edges'].append(edge)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Similarity graph exported to {output_path}")
    
    def clear_cache(self):
        """清空缓存"""
        self.embedding_cache.clear()
        self.similarity_cache.clear()
        logger.info("Cache cleared")