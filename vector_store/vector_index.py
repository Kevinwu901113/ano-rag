import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import faiss
from utils import GPUUtils, FileUtils
from config import config

class VectorIndex:
    """向量索引类，负责构建和管理FAISS索引"""
    
    def __init__(self, embedding_dim: int = None):
        # 配置参数
        self.embedding_dim = embedding_dim or config.get('vector_store.dimension', 768)
        self.index_type = config.get('vector_store.index_type', 'IVFFlat')
        self.similarity_metric = config.get('vector_store.similarity_metric', 'cosine')
        self.use_gpu = config.get('performance.use_gpu', True) and GPUUtils.is_cuda_available()
        
        # 索引参数
        self.nlist = config.get('vector_store.nlist', 100)  # IVF聚类数
        self.nprobe = config.get('vector_store.nprobe', 10)  # 搜索时的聚类数
        self.m = config.get('vector_store.pq_m', 8)  # PQ编码的子向量数
        
        # 索引对象
        self.index = None
        self.is_trained = False
        self.total_vectors = 0
        
        # 存储路径
        self.index_dir = config.get('storage.vector_index_path')
        if not self.index_dir:
            work_dir = config.get('storage.work_dir')
            if work_dir:
                self.index_dir = os.path.join(work_dir, 'vector_index')
            else:
                # 使用临时目录避免在项目根目录创建data文件夹
                import tempfile
                self.index_dir = os.path.join(tempfile.gettempdir(), 'anorag_vector_index')
        FileUtils.ensure_dir(self.index_dir)
        
        # GPU资源
        self.gpu_resource = None
        if self.use_gpu:
            self._setup_gpu_resources()
        
        logger.info(f"VectorIndex initialized: dim={self.embedding_dim}, type={self.index_type}, gpu={self.use_gpu}")
    
    def _setup_gpu_resources(self):
        """设置GPU资源"""
        try:
            if faiss.get_num_gpus() > 0:
                self.gpu_resource = faiss.StandardGpuResources()
                logger.info(f"GPU resources initialized, available GPUs: {faiss.get_num_gpus()}")
            else:
                logger.warning("No GPU available for FAISS")
                self.use_gpu = False
        except Exception as e:
            logger.warning(f"Failed to setup GPU resources: {e}")
            self.use_gpu = False
    
    def create_index(self, index_type: str = None) -> bool:
        """创建向量索引"""
        index_type = index_type or self.index_type
        
        try:
            logger.info(f"Creating {index_type} index with dimension {self.embedding_dim}")
            
            # 根据相似度度量选择距离类型
            if self.similarity_metric == 'cosine':
                # 余弦相似度使用内积（需要归一化向量）
                metric = faiss.METRIC_INNER_PRODUCT
            else:
                # 欧几里得距离
                metric = faiss.METRIC_L2
            
            # 创建不同类型的索引
            if index_type == 'Flat':
                # 暴力搜索，精确但慢
                self.index = faiss.IndexFlatIP(self.embedding_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.embedding_dim)
                self.is_trained = True
                
            elif index_type == 'IVFFlat':
                # 倒排文件索引
                quantizer = faiss.IndexFlatIP(self.embedding_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, metric)
                self.index.nprobe = self.nprobe
                
            elif index_type == 'IVFPQ':
                # 倒排文件 + 乘积量化
                quantizer = faiss.IndexFlatIP(self.embedding_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, self.nlist, self.m, 8, metric)
                self.index.nprobe = self.nprobe
                
            elif index_type == 'HNSW':
                # 分层导航小世界图
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, metric)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
                self.is_trained = True
                
            elif index_type == 'LSH':
                # 局部敏感哈希
                self.index = faiss.IndexLSH(self.embedding_dim, 256)
                self.is_trained = True
                
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # 移动到GPU
            if self.use_gpu and self.gpu_resource is not None:
                try:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)
                    logger.info("Index moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU: {e}")
                    self.use_gpu = False
            
            self.index_type = index_type
            logger.info(f"Index created successfully: {index_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def train_index(self, training_vectors: np.ndarray) -> bool:
        """训练索引（某些索引类型需要）"""
        if self.index is None:
            logger.error("Index not created yet")
            return False
        
        if self.is_trained:
            logger.info("Index already trained")
            return True
        
        try:
            logger.info(f"Training index with {len(training_vectors)} vectors")
            
            # 对于IVF类型的索引，检查训练数据是否足够
            if self.index_type in ['IVFFlat', 'IVFPQ']:
                min_required = self.nlist * 2  # 每个聚类至少需要2个向量
                if len(training_vectors) < min_required:
                    # 动态调整nlist参数
                    new_nlist = max(1, len(training_vectors) // 2)
                    logger.warning(f"Training data insufficient for nlist={self.nlist}, adjusting to {new_nlist}")
                    
                    # 重新创建索引
                    old_index_type = self.index_type
                    old_nlist = self.nlist
                    self.nlist = new_nlist
                    
                    if not self.create_index(old_index_type):
                        # 恢复原参数
                        self.nlist = old_nlist
                        return False
            
            # 预处理训练向量
            processed_vectors = self._preprocess_vectors(training_vectors)
            
            # 训练索引
            self.index.train(processed_vectors)
            self.is_trained = True
            
            logger.info("Index training completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train index: {e}")
            return False
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> bool:
        """添加向量到索引"""
        if self.index is None:
            logger.error("Index not created yet")
            return False
        
        if not self.is_trained and self.index_type in ['IVFFlat', 'IVFPQ']:
            logger.info("Training index with provided vectors")
            if not self.train_index(vectors):
                return False
        
        try:
            # 预处理向量
            processed_vectors = self._preprocess_vectors(vectors)
            
            # 添加向量
            if ids is not None and self.index_type not in ['Flat', 'HNSW', 'LSH']:
                # 只有某些索引类型支持自定义ID
                if hasattr(self.index, 'add_with_ids'):
                    self.index.add_with_ids(processed_vectors, ids)
                else:
                    logger.warning("Index does not support custom IDs, using auto-generated IDs")
                    self.index.add(processed_vectors)
            else:
                # 使用自动生成的ID
                self.index.add(processed_vectors)
            
            self.total_vectors += len(vectors)
            logger.info(f"Added {len(vectors)} vectors to index, total: {self.total_vectors}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            return False
    
    def search(self, query_vectors: np.ndarray, 
              top_k: int = 10,
              return_vectors: bool = False) -> List[Dict[str, Any]]:
        """搜索最相似的向量"""
        if self.index is None:
            logger.error("Index not created yet")
            return []
        
        if self.total_vectors == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # 预处理查询向量
            processed_queries = self._preprocess_vectors(query_vectors)
            
            # 执行搜索
            scores, indices = self.index.search(processed_queries, top_k)
            
            # 处理结果
            results = []
            for query_idx in range(len(processed_queries)):
                query_results = []
                for rank in range(top_k):
                    idx = indices[query_idx][rank]
                    score = scores[query_idx][rank]
                    
                    # 过滤无效结果
                    if idx == -1:
                        continue
                    
                    result = {
                        'index': int(idx),
                        'score': float(score),
                        'rank': rank
                    }
                    
                    # 转换分数为相似度
                    if self.similarity_metric == 'cosine':
                        # 内积分数已经是相似度
                        result['similarity'] = float(score)
                    else:
                        # L2距离转换为相似度
                        result['similarity'] = 1.0 / (1.0 + float(score))
                    
                    query_results.append(result)
                
                results.append(query_results)
            
            # 如果只有一个查询，返回扁平列表
            if len(results) == 1:
                return results[0]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            return []
    
    def _preprocess_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """预处理向量"""
        # 确保是float32类型
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # 确保是连续的内存布局
        if not vectors.flags['C_CONTIGUOUS']:
            vectors = np.ascontiguousarray(vectors)
        
        # 余弦相似度需要归一化
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # 避免除零
            norms = np.where(norms == 0, 1, norms)
            vectors = vectors / norms
        
        return vectors
    
    def save_index(self, filename: str = None) -> str:
        """保存索引到文件"""
        if self.index is None:
            logger.error("No index to save")
            return ""
        
        try:
            filename = filename or f"index_{self.index_type}_{self.embedding_dim}d.faiss"
            filepath = os.path.join(self.index_dir, filename)
            
            # 如果在GPU上，先移动到CPU
            index_to_save = self.index
            if self.use_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            
            # 保存索引
            faiss.write_index(index_to_save, filepath)
            
            # 保存元数据
            metadata = {
                'index_type': self.index_type,
                'embedding_dim': self.embedding_dim,
                'similarity_metric': self.similarity_metric,
                'total_vectors': self.total_vectors,
                'is_trained': self.is_trained,
                'nlist': self.nlist,
                'nprobe': self.nprobe
            }
            
            metadata_file = filepath.replace('.faiss', '_metadata.json')
            FileUtils.write_json(metadata, metadata_file)
            
            logger.info(f"Index saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return ""
    
    def load_index(self, filename: str) -> bool:
        """从文件加载索引"""
        try:
            filepath = os.path.join(self.index_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Index file not found: {filepath}")
                return False
            
            # 加载索引
            self.index = faiss.read_index(filepath)
            
            # 加载元数据
            metadata_file = filepath.replace('.faiss', '_metadata.json')
            if os.path.exists(metadata_file):
                metadata = FileUtils.read_json(metadata_file)
                self.index_type = metadata.get('index_type', self.index_type)
                self.embedding_dim = metadata.get('embedding_dim', self.embedding_dim)
                self.similarity_metric = metadata.get('similarity_metric', self.similarity_metric)
                self.total_vectors = metadata.get('total_vectors', 0)
                self.is_trained = metadata.get('is_trained', True)
                self.nlist = metadata.get('nlist', self.nlist)
                self.nprobe = metadata.get('nprobe', self.nprobe)
            
            # 移动到GPU
            if self.use_gpu and self.gpu_resource is not None:
                try:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)
                    logger.info("Index moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move loaded index to GPU: {e}")
            
            # 设置搜索参数
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            logger.info(f"Index loaded from {filepath}, vectors: {self.total_vectors}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if self.index is None:
            return {}
        
        stats = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric,
            'total_vectors': self.total_vectors,
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu
        }
        
        # 添加索引特定的统计信息
        if hasattr(self.index, 'ntotal'):
            stats['ntotal'] = self.index.ntotal
        
        if hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
            stats['nprobe'] = self.index.nprobe
        
        if hasattr(self.index, 'hnsw'):
            stats['hnsw_M'] = self.index.hnsw.M
            stats['hnsw_efConstruction'] = self.index.hnsw.efConstruction
            stats['hnsw_efSearch'] = self.index.hnsw.efSearch
        
        return stats
    
    def remove_vectors(self, ids: np.ndarray) -> bool:
        """从索引中移除向量（如果支持）"""
        if self.index is None:
            logger.error("Index not created yet")
            return False
        
        try:
            if hasattr(self.index, 'remove_ids'):
                self.index.remove_ids(ids)
                self.total_vectors -= len(ids)
                logger.info(f"Removed {len(ids)} vectors from index")
                return True
            else:
                logger.warning("Index type does not support vector removal")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            return False
    
    def reset_index(self):
        """重置索引"""
        if self.index is not None:
            if hasattr(self.index, 'reset'):
                self.index.reset()
            else:
                # 重新创建索引
                self.create_index(self.index_type)
        
        self.total_vectors = 0
        self.is_trained = False
        logger.info("Index reset completed")
    
    def optimize_search_params(self, query_vectors: np.ndarray, 
                              ground_truth_indices: np.ndarray,
                              target_recall: float = 0.9) -> Dict[str, Any]:
        """优化搜索参数以达到目标召回率"""
        if self.index_type not in ['IVFFlat', 'IVFPQ']:
            logger.warning("Search parameter optimization only supported for IVF indices")
            return {}
        
        best_params = {'nprobe': self.nprobe, 'recall': 0.0}
        
        # 测试不同的nprobe值
        for nprobe in [1, 5, 10, 20, 50, 100]:
            if nprobe > self.nlist:
                break
            
            # 设置参数
            original_nprobe = self.index.nprobe
            self.index.nprobe = nprobe
            
            # 执行搜索
            results = self.search(query_vectors, top_k=len(ground_truth_indices[0]))
            
            # 计算召回率
            recall = self._calculate_recall(results, ground_truth_indices)
            
            logger.info(f"nprobe={nprobe}, recall={recall:.3f}")
            
            if recall > best_params['recall']:
                best_params = {'nprobe': nprobe, 'recall': recall}
            
            # 如果达到目标召回率，停止搜索
            if recall >= target_recall:
                break
            
            # 恢复原始参数
            self.index.nprobe = original_nprobe
        
        # 设置最佳参数
        self.index.nprobe = best_params['nprobe']
        self.nprobe = best_params['nprobe']
        
        logger.info(f"Optimized search params: nprobe={best_params['nprobe']}, recall={best_params['recall']:.3f}")
        return best_params
    
    def _calculate_recall(self, search_results: List[Dict[str, Any]], 
                         ground_truth: np.ndarray) -> float:
        """计算召回率"""
        if not search_results or len(ground_truth) == 0:
            return 0.0
        
        total_recall = 0.0
        
        for i, query_results in enumerate(search_results):
            if i >= len(ground_truth):
                break
            
            retrieved_indices = set(result['index'] for result in query_results)
            true_indices = set(ground_truth[i])
            
            if len(true_indices) > 0:
                recall = len(retrieved_indices & true_indices) / len(true_indices)
                total_recall += recall
        
        return total_recall / min(len(search_results), len(ground_truth))
    
    def cleanup(self):
        """清理资源"""
        if self.gpu_resource is not None:
            # FAISS GPU资源会自动清理
            pass
        
        self.index = None
        logger.info("VectorIndex cleanup completed")