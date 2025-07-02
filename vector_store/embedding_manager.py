import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from utils import BatchProcessor, GPUUtils, FileUtils
from config import config

class EmbeddingManager:
    """嵌入管理器，负责生成和管理文本的向量嵌入"""
    
    def __init__(self):
        # 配置参数
        self.model_name = config.get('embedding.model_name', 'BAAI/bge-m3')
        self.batch_size = config.get('embedding.batch_size', 32)
        self.device = GPUUtils.get_device()
        self.max_length = config.get('embedding.max_length', 512)
        self.normalize_embeddings = config.get('embedding.normalize', True)
        
        # 初始化模型
        self.model = None
        self.embedding_dim = None
        self._load_model()
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=self.batch_size,
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # 缓存路径
        self.cache_dir = config.get('storage.embedding_cache_path', './data/embeddings')
        FileUtils.ensure_dir(self.cache_dir)
        
        logger.info(f"EmbeddingManager initialized with model: {self.model_name}, device: {self.device}")
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # 支持多种模型
            if 'bge-m3' in self.model_name.lower():
                self.model = SentenceTransformer(self.model_name, device=self.device)
                # BGE-M3 支持多语言和多粒度
                self.embedding_dim = 1024
            elif 'nomic-embed' in self.model_name.lower():
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = 768
            else:
                # 通用的sentence-transformers模型
                self.model = SentenceTransformer(self.model_name, device=self.device)
                # 获取实际的嵌入维度
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self.embedding_dim = test_embedding.shape[1]
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length
            
            logger.info(f"Model loaded successfully, embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # 回退到默认模型
            try:
                logger.info("Falling back to default model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.embedding_dim = 384
                self.model_name = 'all-MiniLM-L6-v2'
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError("Cannot load any embedding model")
    
    def encode_texts(self, texts: List[str], 
                    batch_size: Optional[int] = None,
                    show_progress: bool = True,
                    normalize: Optional[bool] = None) -> np.ndarray:
        """编码文本列表为向量嵌入"""
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        normalize = normalize if normalize is not None else self.normalize_embeddings
        
        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
            
            # 预处理文本
            processed_texts = self._preprocess_texts(texts)
            
            # 生成嵌入
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=self.device
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            # 返回零向量作为回退
            return np.zeros((len(texts), self.embedding_dim))
    
    def encode_atomic_notes(self, atomic_notes: List[Dict[str, Any]], 
                           content_field: str = 'content',
                           include_metadata: bool = True) -> np.ndarray:
        """编码原子笔记为向量嵌入"""
        if not atomic_notes:
            return np.array([])
        
        # 提取文本内容
        texts = []
        for note in atomic_notes:
            text_parts = []
            
            # 主要内容
            content = note.get(content_field, '')
            if content:
                text_parts.append(content)
            
            # 包含元数据
            if include_metadata:
                # 关键词
                keywords = note.get('keywords', [])
                if keywords:
                    text_parts.append(f"Keywords: {', '.join(keywords)}")
                
                # 实体
                entities = note.get('entities', [])
                if entities:
                    text_parts.append(f"Entities: {', '.join(entities)}")
                
                # 主题
                topic = note.get('topic', '')
                if topic:
                    text_parts.append(f"Topic: {topic}")
            
            # 合并文本
            full_text = ' '.join(text_parts) if text_parts else 'Empty note'
            texts.append(full_text)
        
        logger.info(f"Encoding {len(atomic_notes)} atomic notes")
        return self.encode_texts(texts)
    
    def encode_queries(self, queries: List[str], 
                      query_prefix: str = "Represent this sentence for searching relevant passages: ") -> np.ndarray:
        """编码查询文本（针对检索优化）"""
        if not queries:
            return np.array([])
        
        # 为查询添加前缀（某些模型需要）
        if 'bge' in self.model_name.lower() and query_prefix:
            prefixed_queries = [query_prefix + query for query in queries]
        else:
            prefixed_queries = queries
        
        logger.info(f"Encoding {len(queries)} queries")
        return self.encode_texts(prefixed_queries)
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """预处理文本"""
        processed = []
        for text in texts:
            # 清理文本
            cleaned = text.strip()
            
            # 截断过长的文本
            if len(cleaned) > self.max_length * 4:  # 粗略估计token数
                cleaned = cleaned[:self.max_length * 4]
                logger.debug(f"Truncated text from {len(text)} to {len(cleaned)} characters")
            
            # 处理空文本
            if not cleaned:
                cleaned = "Empty content"
            
            processed.append(cleaned)
        
        return processed
    
    def compute_similarity(self, embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          metric: str = 'cosine') -> np.ndarray:
        """计算嵌入之间的相似度"""
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([])
        
        try:
            if metric == 'cosine':
                # 余弦相似度
                if embeddings1.ndim == 1:
                    embeddings1 = embeddings1.reshape(1, -1)
                if embeddings2.ndim == 1:
                    embeddings2 = embeddings2.reshape(1, -1)
                
                # 归一化
                norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
                
                embeddings1_norm = embeddings1 / (norm1 + 1e-8)
                embeddings2_norm = embeddings2 / (norm2 + 1e-8)
                
                # 计算相似度
                similarity = np.dot(embeddings1_norm, embeddings2_norm.T)
                
            elif metric == 'euclidean':
                # 欧几里得距离（转换为相似度）
                from scipy.spatial.distance import cdist
                distances = cdist(embeddings1, embeddings2, metric='euclidean')
                # 转换为相似度（距离越小，相似度越高）
                similarity = 1 / (1 + distances)
                
            elif metric == 'dot':
                # 点积相似度
                similarity = np.dot(embeddings1, embeddings2.T)
                
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return np.array([])
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 10,
                         metric: str = 'cosine') -> List[Dict[str, Any]]:
        """找到最相似的嵌入"""
        if query_embedding.size == 0 or candidate_embeddings.size == 0:
            return []
        
        # 计算相似度
        similarities = self.compute_similarity(
            query_embedding.reshape(1, -1), 
            candidate_embeddings, 
            metric=metric
        )
        
        if similarities.size == 0:
            return []
        
        # 获取top-k结果
        similarities = similarities.flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       metadata: Dict[str, Any],
                       filename: str) -> str:
        """保存嵌入到文件"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            
            # 保存数据
            np.savez_compressed(
                filepath,
                embeddings=embeddings,
                metadata=metadata,
                model_name=self.model_name,
                embedding_dim=self.embedding_dim
            )
            
            logger.info(f"Embeddings saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return ""
    
    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        """从文件加载嵌入"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Embedding file not found: {filepath}")
                return {}
            
            data = np.load(filepath, allow_pickle=True)
            
            result = {
                'embeddings': data['embeddings'],
                'metadata': data['metadata'].item() if 'metadata' in data else {},
                'model_name': str(data['model_name']) if 'model_name' in data else '',
                'embedding_dim': int(data['embedding_dim']) if 'embedding_dim' in data else 0
            }
            
            logger.info(f"Embeddings loaded from {filepath}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return {}
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """获取嵌入统计信息"""
        if embeddings.size == 0:
            return {}
        
        stats = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'memory_mb': embeddings.nbytes / (1024 * 1024)
        }
        
        return stats
    
    def batch_encode_with_cache(self, texts: List[str], 
                               cache_key: str,
                               force_recompute: bool = False) -> np.ndarray:
        """带缓存的批量编码"""
        cache_file = f"{cache_key}_embeddings.npz"
        
        # 检查缓存
        if not force_recompute:
            cached_data = self.load_embeddings(cache_file)
            if cached_data and 'embeddings' in cached_data:
                cached_embeddings = cached_data['embeddings']
                if cached_embeddings.shape[0] == len(texts):
                    logger.info(f"Using cached embeddings for {cache_key}")
                    return cached_embeddings
        
        # 生成新的嵌入
        logger.info(f"Computing new embeddings for {cache_key}")
        embeddings = self.encode_texts(texts)
        
        # 保存到缓存
        metadata = {
            'text_count': len(texts),
            'cache_key': cache_key,
            'timestamp': self._get_timestamp()
        }
        self.save_embeddings(embeddings, metadata, cache_file)
        
        return embeddings
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'normalize_embeddings': self.normalize_embeddings
        }
    
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("EmbeddingManager cleanup completed")