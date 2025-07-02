import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from .embedding_manager import EmbeddingManager
from .vector_index import VectorIndex
from utils import BatchProcessor, FileUtils
from config import config

class VectorRetriever:
    """向量检索器，整合嵌入管理和向量索引功能"""
    
    def __init__(self):
        # 初始化组件
        self.embedding_manager = EmbeddingManager()
        self.vector_index = VectorIndex(self.embedding_manager.embedding_dim)
        
        # 配置参数
        self.top_k = config.get('vector_store.top_k', 20)
        self.similarity_threshold = config.get('vector_store.similarity_threshold', 0.5)
        self.batch_size = config.get('vector_store.batch_size', 32)
        
        # 数据存储
        self.atomic_notes = []  # 存储原子笔记
        self.note_embeddings = None  # 存储嵌入
        self.note_id_to_index = {}  # 笔记ID到索引的映射
        self.index_to_note_id = {}  # 索引到笔记ID的映射
        
        # 存储路径
        self.data_dir = config.get('storage.vector_store_path', './data/vector_store')
        FileUtils.ensure_dir(self.data_dir)
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=self.batch_size,
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        logger.info("VectorRetriever initialized")
    
    def build_index(self, atomic_notes: List[Dict[str, Any]], 
                   force_rebuild: bool = False,
                   save_index: bool = True) -> bool:
        """构建向量索引"""
        if not atomic_notes:
            logger.warning("No atomic notes provided for indexing")
            return False
        
        try:
            logger.info(f"Building vector index for {len(atomic_notes)} atomic notes")
            
            # 检查是否需要重建
            if not force_rebuild and self._can_load_existing_index(atomic_notes):
                logger.info("Loading existing index")
                return True
            
            # 存储原子笔记
            self.atomic_notes = atomic_notes
            self._build_id_mappings()
            
            # 生成嵌入
            logger.info("Generating embeddings for atomic notes")
            self.note_embeddings = self.embedding_manager.encode_atomic_notes(
                atomic_notes, 
                include_metadata=True
            )
            
            if self.note_embeddings.size == 0:
                logger.error("Failed to generate embeddings")
                return False
            
            # 创建向量索引
            logger.info("Creating vector index")
            if not self.vector_index.create_index():
                logger.error("Failed to create vector index")
                return False
            
            # 添加向量到索引
            note_ids = np.array([i for i in range(len(atomic_notes))], dtype=np.int64)
            if not self.vector_index.add_vectors(self.note_embeddings, note_ids):
                logger.error("Failed to add vectors to index")
                return False
            
            # 保存索引和数据
            if save_index:
                self._save_index_data()
            
            logger.info(f"Vector index built successfully with {len(atomic_notes)} notes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            return False
    
    def search(self, queries: List[str], 
              top_k: Optional[int] = None,
              similarity_threshold: Optional[float] = None,
              include_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """搜索相似的原子笔记"""
        if not queries:
            return []
        
        if not self.atomic_notes or self.vector_index.total_vectors == 0:
            logger.warning("Vector index is empty")
            return [[] for _ in queries]
        
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            logger.info(f"Searching for {len(queries)} queries, top_k={top_k}")
            
            # 生成查询嵌入
            query_embeddings = self.embedding_manager.encode_queries(queries)
            
            if query_embeddings.size == 0:
                logger.error("Failed to generate query embeddings")
                return [[] for _ in queries]
            
            # 执行向量搜索
            search_results = self.vector_index.search(
                query_embeddings, 
                top_k=top_k
            )
            
            # 处理搜索结果
            final_results = []
            
            # 确保search_results是列表的列表
            if len(queries) == 1 and isinstance(search_results, list) and \
               len(search_results) > 0 and isinstance(search_results[0], dict):
                search_results = [search_results]
            
            for query_idx, query in enumerate(queries):
                query_results = []
                
                if query_idx < len(search_results):
                    for result in search_results[query_idx]:
                        # 过滤低相似度结果
                        if result.get('similarity', 0) < similarity_threshold:
                            continue
                        
                        # 获取原子笔记
                        note_index = result['index']
                        if note_index < len(self.atomic_notes):
                            note = self.atomic_notes[note_index].copy()
                            
                            # 添加检索信息
                            retrieval_info = {
                                'similarity': result['similarity'],
                                'score': result['score'],
                                'rank': result['rank'],
                                'query': query,
                                'retrieval_method': 'vector_search'
                            }
                            
                            if include_metadata:
                                note['retrieval_info'] = retrieval_info
                            else:
                                # 只保留核心信息
                                note = {
                                    'note_id': note.get('note_id'),
                                    'content': note.get('content'),
                                    'retrieval_info': retrieval_info
                                }
                            
                            query_results.append(note)
                
                final_results.append(query_results)
            
            # 记录搜索统计
            total_results = sum(len(results) for results in final_results)
            logger.info(f"Search completed: {total_results} results for {len(queries)} queries")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return [[] for _ in queries]
    
    def search_single(self, query: str, 
                     top_k: Optional[int] = None,
                     similarity_threshold: Optional[float] = None,
                     include_metadata: bool = True) -> List[Dict[str, Any]]:
        """搜索单个查询"""
        results = self.search([query], top_k, similarity_threshold, include_metadata)
        return results[0] if results else []
    
    def add_notes(self, new_notes: List[Dict[str, Any]], 
                 rebuild_index: bool = False) -> bool:
        """添加新的原子笔记"""
        if not new_notes:
            return True
        
        try:
            logger.info(f"Adding {len(new_notes)} new notes to index")
            
            if rebuild_index or not self.atomic_notes:
                # 重建整个索引
                all_notes = self.atomic_notes + new_notes
                return self.build_index(all_notes, force_rebuild=True)
            else:
                # 增量添加
                start_index = len(self.atomic_notes)
                self.atomic_notes.extend(new_notes)
                
                # 更新ID映射
                self._build_id_mappings()
                
                # 生成新笔记的嵌入
                new_embeddings = self.embedding_manager.encode_atomic_notes(
                    new_notes, 
                    include_metadata=True
                )
                
                if new_embeddings.size == 0:
                    logger.error("Failed to generate embeddings for new notes")
                    return False
                
                # 添加到索引
                new_ids = np.array([i for i in range(start_index, start_index + len(new_notes))], dtype=np.int64)
                if not self.vector_index.add_vectors(new_embeddings, new_ids):
                    logger.error("Failed to add new vectors to index")
                    return False
                
                # 更新嵌入矩阵
                if self.note_embeddings is not None:
                    self.note_embeddings = np.vstack([self.note_embeddings, new_embeddings])
                else:
                    self.note_embeddings = new_embeddings
                
                logger.info(f"Successfully added {len(new_notes)} notes to index")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add notes: {e}")
            return False
    
    def remove_notes(self, note_ids: List[str]) -> bool:
        """移除指定的原子笔记"""
        if not note_ids:
            return True
        
        try:
            logger.info(f"Removing {len(note_ids)} notes from index")
            
            # 找到要移除的索引
            indices_to_remove = []
            for note_id in note_ids:
                if note_id in self.note_id_to_index:
                    indices_to_remove.append(self.note_id_to_index[note_id])
            
            if not indices_to_remove:
                logger.warning("No matching notes found to remove")
                return True
            
            # 由于FAISS索引的限制，我们需要重建索引
            logger.info("Rebuilding index after note removal")
            
            # 移除笔记
            indices_to_remove = sorted(indices_to_remove, reverse=True)
            for idx in indices_to_remove:
                if idx < len(self.atomic_notes):
                    del self.atomic_notes[idx]
            
            # 重建索引
            return self.build_index(self.atomic_notes, force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Failed to remove notes: {e}")
            return False
    
    def update_note(self, note_id: str, updated_note: Dict[str, Any]) -> bool:
        """更新指定的原子笔记"""
        try:
            if note_id not in self.note_id_to_index:
                logger.warning(f"Note {note_id} not found")
                return False
            
            # 更新笔记
            note_index = self.note_id_to_index[note_id]
            self.atomic_notes[note_index] = updated_note
            
            # 重新生成该笔记的嵌入
            new_embedding = self.embedding_manager.encode_atomic_notes(
                [updated_note], 
                include_metadata=True
            )
            
            if new_embedding.size == 0:
                logger.error("Failed to generate embedding for updated note")
                return False
            
            # 更新嵌入矩阵
            if self.note_embeddings is not None:
                self.note_embeddings[note_index] = new_embedding[0]
            
            # 由于FAISS的限制，需要重建索引
            logger.info(f"Rebuilding index after updating note {note_id}")
            return self.build_index(self.atomic_notes, force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            return False
    
    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取原子笔记"""
        if note_id in self.note_id_to_index:
            note_index = self.note_id_to_index[note_id]
            if note_index < len(self.atomic_notes):
                return self.atomic_notes[note_index]
        return None
    
    def get_notes_by_ids(self, note_ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID列表获取原子笔记"""
        notes = []
        for note_id in note_ids:
            note = self.get_note_by_id(note_id)
            if note:
                notes.append(note)
        return notes
    
    def get_similar_notes(self, note_id: str, 
                         top_k: int = 10,
                         exclude_self: bool = True) -> List[Dict[str, Any]]:
        """获取与指定笔记相似的其他笔记"""
        note = self.get_note_by_id(note_id)
        if not note:
            return []
        
        # 使用笔记内容进行搜索
        content = note.get('content', '')
        if not content:
            return []
        
        results = self.search_single(content, top_k=top_k + (1 if exclude_self else 0))
        
        # 排除自身
        if exclude_self:
            results = [r for r in results if r.get('note_id') != note_id]
        
        return results[:top_k]
    
    def _build_id_mappings(self):
        """构建ID映射"""
        self.note_id_to_index = {}
        self.index_to_note_id = {}
        
        for idx, note in enumerate(self.atomic_notes):
            note_id = note.get('note_id')
            if note_id:
                self.note_id_to_index[note_id] = idx
                self.index_to_note_id[idx] = note_id
    
    def _can_load_existing_index(self, atomic_notes: List[Dict[str, Any]]) -> bool:
        """检查是否可以加载现有索引"""
        # 检查索引文件是否存在
        index_files = [f for f in os.listdir(self.data_dir) if f.endswith('.faiss')]
        if not index_files:
            return False
        
        # 检查数据文件是否存在
        data_file = os.path.join(self.data_dir, 'atomic_notes.json')
        if not os.path.exists(data_file):
            return False
        
        # 加载现有数据
        try:
            existing_notes = FileUtils.read_json(data_file)
            
            # 简单比较：检查笔记数量和第一个笔记的ID
            if len(existing_notes) != len(atomic_notes):
                return False
            
            if existing_notes and atomic_notes:
                if existing_notes[0].get('note_id') != atomic_notes[0].get('note_id'):
                    return False
            
            # 加载索引
            index_file = index_files[0]
            if self.vector_index.load_index(index_file):
                self.atomic_notes = existing_notes
                self._build_id_mappings()
                
                # 加载嵌入
                embedding_file = os.path.join(self.data_dir, 'note_embeddings.npz')
                if os.path.exists(embedding_file):
                    data = np.load(embedding_file)
                    self.note_embeddings = data['embeddings']
                
                logger.info(f"Loaded existing index with {len(self.atomic_notes)} notes")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
        
        return False
    
    def _save_index_data(self):
        """保存索引和数据"""
        try:
            # 保存向量索引
            self.vector_index.save_index()
            
            # 保存原子笔记
            notes_file = os.path.join(self.data_dir, 'atomic_notes.json')
            FileUtils.write_json(self.atomic_notes, notes_file)
            
            # 保存嵌入
            if self.note_embeddings is not None:
                embedding_file = os.path.join(self.data_dir, 'note_embeddings.npz')
                np.savez_compressed(embedding_file, embeddings=self.note_embeddings)
            
            # 保存ID映射
            mapping_file = os.path.join(self.data_dir, 'id_mappings.json')
            FileUtils.write_json({
                'note_id_to_index': self.note_id_to_index,
                'index_to_note_id': self.index_to_note_id
            }, mapping_file)
            
            logger.info("Index data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save index data: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        stats = {
            'total_notes': len(self.atomic_notes),
            'embedding_dim': self.embedding_manager.embedding_dim,
            'model_name': self.embedding_manager.model_name,
            'index_stats': self.vector_index.get_index_stats(),
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }
        
        if self.note_embeddings is not None:
            stats['embedding_stats'] = self.embedding_manager.get_embedding_stats(self.note_embeddings)
        
        return stats
    
    def optimize_retrieval(self, test_queries: List[str], 
                          ground_truth: List[List[str]],
                          target_recall: float = 0.9) -> Dict[str, Any]:
        """优化检索参数"""
        if not test_queries or not ground_truth:
            logger.warning("No test data provided for optimization")
            return {}
        
        logger.info(f"Optimizing retrieval parameters with {len(test_queries)} test queries")
        
        # 转换ground truth为索引
        gt_indices = []
        for gt_note_ids in ground_truth:
            indices = []
            for note_id in gt_note_ids:
                if note_id in self.note_id_to_index:
                    indices.append(self.note_id_to_index[note_id])
            gt_indices.append(indices)
        
        # 生成测试查询的嵌入
        query_embeddings = self.embedding_manager.encode_queries(test_queries)
        
        # 优化向量索引参数
        index_optimization = self.vector_index.optimize_search_params(
            query_embeddings, 
            np.array(gt_indices, dtype=object),
            target_recall
        )
        
        # 测试不同的相似度阈值
        best_threshold = self.similarity_threshold
        best_f1 = 0.0
        
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # 执行搜索
            results = self.search(test_queries, similarity_threshold=threshold)
            
            # 计算F1分数
            f1_score = self._calculate_f1_score(results, ground_truth)
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
        
        # 更新最佳参数
        self.similarity_threshold = best_threshold
        
        optimization_result = {
            'index_optimization': index_optimization,
            'best_similarity_threshold': best_threshold,
            'best_f1_score': best_f1,
            'target_recall': target_recall
        }
        
        logger.info(f"Optimization completed: threshold={best_threshold}, F1={best_f1:.3f}")
        return optimization_result
    
    def _calculate_f1_score(self, search_results: List[List[Dict[str, Any]]], 
                           ground_truth: List[List[str]]) -> float:
        """计算F1分数"""
        if not search_results or not ground_truth:
            return 0.0
        
        total_f1 = 0.0
        valid_queries = 0
        
        for i, (results, gt) in enumerate(zip(search_results, ground_truth)):
            if not gt:
                continue
            
            retrieved_ids = set(result.get('note_id') for result in results)
            true_ids = set(gt)
            
            if len(retrieved_ids) == 0 and len(true_ids) == 0:
                f1 = 1.0
            elif len(retrieved_ids) == 0 or len(true_ids) == 0:
                f1 = 0.0
            else:
                precision = len(retrieved_ids & true_ids) / len(retrieved_ids)
                recall = len(retrieved_ids & true_ids) / len(true_ids)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            total_f1 += f1
            valid_queries += 1
        
        return total_f1 / valid_queries if valid_queries > 0 else 0.0
    
    def clear_index(self):
        """清空索引"""
        self.atomic_notes = []
        self.note_embeddings = None
        self.note_id_to_index = {}
        self.index_to_note_id = {}
        
        if self.vector_index:
            self.vector_index.reset_index()
        
        logger.info("Vector index cleared")
    
    def cleanup(self):
        """清理资源"""
        if self.embedding_manager:
            self.embedding_manager.cleanup()
        
        if self.vector_index:
            self.vector_index.cleanup()
        
        logger.info("VectorRetriever cleanup completed")