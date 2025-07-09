import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from utils import BatchProcessor, GPUUtils, FileUtils
from config import config

class EmbeddingManager:
    """嵌入管理器，专注于本地模型加载"""
    
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
        self._load_local_model()
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=self.batch_size,
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # 缓存路径
        self.cache_dir = config.get('storage.embedding_cache_path')
        if not self.cache_dir:
            work_dir = config.get('storage.work_dir')
            if work_dir:
                self.cache_dir = os.path.join(work_dir, 'embeddings')
            else:
                # 使用临时目录避免在项目根目录创建data文件夹
                import tempfile
                self.cache_dir = os.path.join(tempfile.gettempdir(), 'anorag_embeddings')
        FileUtils.ensure_dir(self.cache_dir)
        
        logger.info(f"EmbeddingManager initialized with model: {self.model_name}, device: {self.device}")
    
    def _load_local_model(self):
        """专门加载本地模型"""
        try:
            # 设置离线模式，避免网络请求
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # 获取本地模型目录
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            local_models_dir = os.path.join(repo_root, 'models/embedding')
            
            # 定义本地模型路径映射
            local_model_paths = self._get_local_model_paths(local_models_dir)
            
            # 尝试加载本地模型
            for model_path in local_model_paths:
                if self._try_load_model_from_path(model_path):
                    return
            
            # 如果所有路径都失败，抛出异常
            raise RuntimeError(f"无法从本地路径加载任何嵌入模型。检查的路径: {local_model_paths}")
            
        except Exception as e:
            logger.error(f"加载本地嵌入模型失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型: {e}")
    
    def _get_local_model_paths(self, local_models_dir: str) -> List[str]:
        """获取所有可能的本地模型路径"""
        paths = []
        
        # 1. 直接模型名称路径
        paths.append(os.path.join(local_models_dir, self.model_name))
        
        # 2. 替换斜杠的路径
        safe_name = self.model_name.replace('/', '_')
        paths.append(os.path.join(local_models_dir, safe_name))
        
        # 3. HuggingFace缓存格式路径
        hf_cache_name = f"models--{self.model_name.replace('/', '--')}"
        paths.append(os.path.join(local_models_dir, hf_cache_name))
        
        # 4. 特殊处理BAAI/bge-m3模型的snapshots路径
        if self.model_name == 'BAAI/bge-m3':
            bge_cache_path = os.path.join(local_models_dir, 'BAAI_bge-m3')
            snapshots_dir = os.path.join(bge_cache_path, 'snapshots')
            if os.path.isdir(snapshots_dir):
                # 查找所有snapshot目录
                try:
                     snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                    if os.path.isdir(os.path.join(snapshots_dir, d))]
                     if snapshot_dirs:
                         # 查找有实际文件的snapshot目录
                         for snapshot_dir in sorted(snapshot_dirs, reverse=True):  # 从最新开始
                             snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
                             # 检查是否有模型文件
                             if (os.path.exists(os.path.join(snapshot_path, 'modules.json')) or 
                                 os.path.exists(os.path.join(snapshot_path, 'config.json'))):
                                 paths.insert(0, snapshot_path)  # 优先使用
                                 break
                except Exception as e:
                    logger.warning(f"检查snapshots目录失败: {e}")
        
        # 5. 添加sentence-transformers格式的路径
        if 'sentence-transformers' in self.model_name:
            model_short_name = self.model_name.split('/')[-1]
            paths.append(os.path.join(local_models_dir, model_short_name))
        
        # 6. 添加all-MiniLM-L6-v2的HuggingFace缓存路径
        if 'all-MiniLM-L6-v2' in self.model_name:
            minilm_cache_path = os.path.join(local_models_dir, 'models--sentence-transformers--all-MiniLM-L6-v2')
            snapshots_dir = os.path.join(minilm_cache_path, 'snapshots')
            if os.path.isdir(snapshots_dir):
                try:
                     snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                    if os.path.isdir(os.path.join(snapshots_dir, d))]
                     if snapshot_dirs:
                         # 查找有实际文件的snapshot目录
                         for snapshot_dir in sorted(snapshot_dirs, reverse=True):  # 从最新开始
                             snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
                             # 检查是否有模型文件
                             if (os.path.exists(os.path.join(snapshot_path, 'modules.json')) or 
                                 os.path.exists(os.path.join(snapshot_path, 'config.json'))):
                                 paths.insert(0, snapshot_path)  # 优先使用
                                 break
                except Exception as e:
                    logger.warning(f"检查MiniLM snapshots目录失败: {e}")
        
        return paths
    
    def _try_load_model_from_path(self, model_path: str) -> bool:
        """尝试从指定路径加载模型"""
        logger.info(f"尝试从路径加载模型: {model_path}")
        
        if not os.path.isdir(model_path):
            logger.debug(f"路径不存在或不是目录: {model_path}")
            return False
        
        try:
            # 检查必要的模型文件是否存在
            # 对于sentence-transformers模型，检查特有的配置文件
            required_files = ['modules.json', 'config_sentence_transformers.json']
            has_required_files = False
            
            # 检查是否有sentence-transformers的配置文件
            for file in required_files:
                if os.path.exists(os.path.join(model_path, file)):
                    has_required_files = True
                    break
            
            # 如果没有sentence-transformers配置文件，检查通用的config.json
            if not has_required_files and os.path.exists(os.path.join(model_path, 'config.json')):
                has_required_files = True
            
            if not has_required_files:
                logger.debug(f"缺少必要的模型配置文件在路径 {model_path}")
                return False
            
            # 尝试加载模型
            logger.info(f"正在从 {model_path} 加载SentenceTransformer模型")
            
            # 根据模型类型使用不同的加载参数
            load_kwargs = {
                'device': self.device,
                'trust_remote_code': True  # 允许自定义代码
            }
            
            self.model = SentenceTransformer(model_path, **load_kwargs)
            
            # 获取嵌入维度
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # 通过测试编码获取维度
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self.embedding_dim = test_embedding.shape[1]
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length
            
            # 更新模型名称为实际路径
            self.model_name = model_path
            
            logger.info(f"模型加载成功! 路径: {model_path}, 嵌入维度: {self.embedding_dim}")
            return True
            
        except Exception as e:
            logger.warning(f"从路径 {model_path} 加载模型失败: {e}")
            return False
    
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
            logger.info(f"编码 {len(texts)} 个文本，批次大小: {batch_size}")
            
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
            
            logger.info(f"生成嵌入形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"编码文本失败: {e}")
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
        
        logger.info(f"编码 {len(atomic_notes)} 个原子笔记")
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
        
        logger.info(f"编码 {len(queries)} 个查询")
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
                logger.debug(f"文本从 {len(text)} 字符截断到 {len(cleaned)} 字符")
            
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
                raise ValueError(f"不支持的相似度度量: {metric}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
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
        
        logger.info("EmbeddingManager 清理完成")