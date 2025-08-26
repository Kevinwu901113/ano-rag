import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
# 避免循环导入，直接导入需要的模块
try:
    from utils.batch_processor import BatchProcessor
    from utils.gpu_utils import GPUUtils
    from utils.file_utils import FileUtils
except ImportError:
    # 如果导入失败，定义简单的替代类
    class BatchProcessor:
        @staticmethod
        def process_in_batches(items, batch_size, process_func):
            for i in range(0, len(items), batch_size):
                yield process_func(items[i:i+batch_size])
    
    class GPUUtils:
        @staticmethod
        def get_optimal_device():
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class FileUtils:
        @staticmethod
        def ensure_dir(path):
            os.makedirs(path, exist_ok=True)

# 导入模型一致性模块
try:
    from utils.model_consistency import (
        ModelConsistencyChecker, ModelSignature, 
        create_model_consistency_checker, create_model_signature
    )
    MODEL_CONSISTENCY_AVAILABLE = True
except ImportError:
    MODEL_CONSISTENCY_AVAILABLE = False
    logger.warning("Model consistency module not available")

try:
    from config import config
except ImportError:
    # 如果config导入失败，使用默认配置
    config = {
        'embedding': {
            'model_name': 'BAAI/bge-m3',
            'device': 'cpu',
            'batch_size': 32,
            'max_length': 512,
            'normalize': True
        }
    }

class EmbeddingManager:
    """嵌入管理器，专注于本地模型加载"""
    
    _instance = None
    _model_loaded = False
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            import threading
            cls._lock = threading.Lock()
        return cls._instance
    
    def __init__(self):
        # 如果模型已经加载，直接返回
        if self._model_loaded:
            return
            
        with self._lock:
            # 双重检查锁定模式
            if self._model_loaded:
                return
                
            # 配置参数
            self.model_name = config.get('embedding.model_name', 'BAAI/bge-m3')
            self.batch_size = config.get('embedding.batch_size', 32)
            self.device = GPUUtils.get_optimal_device()
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
            
            # 初始化模型一致性检查器
            self.consistency_checker = None
            if MODEL_CONSISTENCY_AVAILABLE:
                try:
                    self.consistency_checker = create_model_consistency_checker(config)
                    logger.info("Model consistency checker initialized")
                    
                    # 注册模型签名
                    if self.model is not None:
                        self.register_model_signature()
                except Exception as e:
                    logger.warning(f"Failed to initialize model consistency checker: {e}")
            
            logger.info(f"EmbeddingManager initialized with model: {self.model_name}, device: {self.device}")
            
            # 标记模型已加载
            EmbeddingManager._model_loaded = True
    
    def _load_local_model(self):
        """专门加载本地模型，如果本地不存在则自动下载"""
        try:
            # 获取本地模型目录
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            local_models_dir = os.path.join(repo_root, 'models/embedding')
            
            # 首先检查本地模型是否存在
            local_model_paths = self._get_local_model_paths(local_models_dir)
            model_exists = any(os.path.exists(path) for path in local_model_paths)
            
            if model_exists:
                logger.info("检测到本地模型，尝试加载...")
                # 设置离线模式，避免网络请求
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
                
                # 尝试加载本地模型
                for model_path in local_model_paths:
                    if self._try_load_model_from_path(model_path):
                        return
                
                logger.warning("本地模型存在但加载失败，尝试重新下载...")
            else:
                logger.info("未检测到本地模型，开始自动下载...")
            
            # 如果本地模型不存在或加载失败，尝试下载
            self._download_and_load_model(local_models_dir)
            
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型: {e}")
    
    def _download_and_load_model(self, local_models_dir: str):
        """下载并加载模型"""
        try:
            # 清除离线模式环境变量，允许网络请求
            if 'TRANSFORMERS_OFFLINE' in os.environ:
                del os.environ['TRANSFORMERS_OFFLINE']
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']
            
            logger.info(f"开始下载模型: {self.model_name}")
            
            # 确保模型目录存在
            os.makedirs(local_models_dir, exist_ok=True)
            
            # 直接使用SentenceTransformer下载模型
            # 这会自动下载到HuggingFace缓存目录
            logger.info("正在从HuggingFace下载模型，这可能需要几分钟...")
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # 保存模型信息
            self.model = model
            self.embedding_dim = model.get_sentence_embedding_dimension()
            self.max_seq_length = getattr(model, 'max_seq_length', 512)
            
            logger.info(f"模型下载并加载成功: {self.model_name}")
            logger.info(f"嵌入维度: {self.embedding_dim}, 最大序列长度: {self.max_seq_length}")
            
            # 可选：将模型保存到本地目录以便下次使用
            self._save_model_locally(model, local_models_dir)
            
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            # 如果下载失败，尝试使用备用模型
            self._try_fallback_model()
    
    def _save_model_locally(self, model, local_models_dir: str):
        """将下载的模型保存到本地目录"""
        try:
            # 创建本地模型路径
            safe_name = self.model_name.replace('/', '_')
            local_model_path = os.path.join(local_models_dir, safe_name)
            
            logger.info(f"保存模型到本地: {local_model_path}")
            model.save(local_model_path)
            logger.info("模型保存成功")
            
        except Exception as e:
            logger.warning(f"保存模型到本地失败: {e}，但模型已成功加载")
    
    def _try_fallback_model(self):
        """尝试使用备用模型"""
        fallback_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        logger.info("尝试使用备用模型...")
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"尝试加载备用模型: {fallback_model}")
                model = SentenceTransformer(
                    fallback_model,
                    device=self.device,
                    trust_remote_code=True
                )
                
                # 保存模型信息
                self.model = model
                self.embedding_dim = model.get_sentence_embedding_dimension()
                self.max_seq_length = getattr(model, 'max_seq_length', 512)
                self.model_name = fallback_model  # 更新模型名称
                
                logger.info(f"备用模型加载成功: {fallback_model}")
                return
                
            except Exception as e:
                logger.warning(f"备用模型 {fallback_model} 加载失败: {e}")
                continue
        
        # 如果所有备用模型都失败，抛出异常
        raise RuntimeError("所有模型（包括备用模型）都无法加载")
    
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
        """编码原子笔记为向量嵌入，使用配置化的文本策略"""
        if not atomic_notes:
            return np.array([])
        
        # 获取嵌入策略配置
        embedding_config = config.get('embedding_strategy', {}).get('atomic_note_embedding', {})
        text_strategy = embedding_config.get('text_strategy', 'title_raw_span')
        field_priority = embedding_config.get('field_priority', ['title', 'raw_span', 'original_text', 'content'])
        text_combination = embedding_config.get('text_combination', {})
        preprocessing = embedding_config.get('preprocessing', {})
        quality_control = embedding_config.get('quality_control', {})
        
        # 提取文本内容
        texts = []
        skipped_count = 0
        
        for i, note in enumerate(atomic_notes):
            try:
                # 根据策略提取文本
                if text_strategy == 'title_raw_span':
                    text = self._extract_title_raw_span_text(note, field_priority, text_combination)
                elif text_strategy == 'content_only':
                    text = note.get(content_field, '')
                elif text_strategy == 'title_content':
                    text = self._extract_title_content_text(note, content_field, text_combination)
                else:  # custom or fallback
                    text = self._extract_title_raw_span_text(note, field_priority, text_combination)
                
                # 文本预处理
                text = self._preprocess_embedding_text(text, preprocessing)
                
                # 质量控制
                if self._should_skip_note(text, quality_control):
                    if quality_control.get('log_skipped_notes', True):
                        logger.debug(f"跳过笔记 {i}: 文本质量不符合要求")
                    skipped_count += 1
                    text = "Empty note"  # 使用占位符
                
                texts.append(text)
                
            except Exception as e:
                logger.warning(f"处理笔记 {i} 时出错: {e}")
                if quality_control.get('skip_invalid_encoding', True):
                    skipped_count += 1
                    texts.append("Empty note")
                else:
                    texts.append(note.get(content_field, 'Empty note'))
        
        if skipped_count > 0:
            logger.info(f"编码 {len(atomic_notes)} 个原子笔记，跳过 {skipped_count} 个，使用策略: {text_strategy}")
        else:
            logger.info(f"编码 {len(atomic_notes)} 个原子笔记，使用策略: {text_strategy}")
        
        return self.encode_texts(texts)
    
    def _extract_title_raw_span_text(self, note: Dict[str, Any], field_priority: List[str], 
                                   text_combination: Dict[str, Any]) -> str:
        """提取title+raw_span组合文本"""
        text_parts = []
        separator = text_combination.get('separator', ' ')
        max_length = text_combination.get('max_combined_length', 512)
        enable_dedup = text_combination.get('enable_deduplication', True)
        
        # 按优先级提取字段
        for field in field_priority:
            value = note.get(field, '').strip()
            if value:
                if not enable_dedup or value not in text_parts:
                    text_parts.append(value)
        
        # 组合文本
        combined_text = separator.join(text_parts)
        
        # 截断处理
        if len(combined_text) > max_length:
            truncate_strategy = text_combination.get('truncate_strategy', 'tail')
            if truncate_strategy == 'head':
                combined_text = combined_text[:max_length]
            elif truncate_strategy == 'tail':
                combined_text = combined_text[-max_length:]
            elif truncate_strategy == 'middle':
                half = max_length // 2
                combined_text = combined_text[:half] + combined_text[-half:]
        
        return combined_text or "Empty note"
    
    def _extract_title_content_text(self, note: Dict[str, Any], content_field: str, 
                                  text_combination: Dict[str, Any]) -> str:
        """提取title+content组合文本"""
        text_parts = []
        separator = text_combination.get('separator', ' ')
        
        title = note.get('title', '').strip()
        content = note.get(content_field, '').strip()
        
        if title:
            text_parts.append(title)
        if content:
            text_parts.append(content)
        
        return separator.join(text_parts) or "Empty note"
    
    def _preprocess_embedding_text(self, text: str, preprocessing: Dict[str, Any]) -> str:
        """预处理嵌入文本"""
        if not text:
            return text
        
        # 移除多余空白
        if preprocessing.get('remove_extra_whitespace', True):
            import re
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Unicode标准化
        if preprocessing.get('normalize_unicode', True):
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        # 移除控制字符
        if preprocessing.get('remove_control_chars', True):
            import re
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _should_skip_note(self, text: str, quality_control: Dict[str, Any]) -> bool:
        """判断是否应该跳过笔记"""
        # 检查空笔记
        if quality_control.get('skip_empty_notes', True) and not text.strip():
            return True
        
        # 检查最小长度
        min_length = quality_control.get('min_text_length', 3)
        if len(text.strip()) < min_length:
            return True
        
        return False
    
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
        info = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'normalize_embeddings': self.normalize_embeddings
        }
        
        # 添加模型一致性信息
        if self.consistency_checker:
            try:
                signature = self.get_model_signature()
                info['consistency_check'] = {
                    'signature': signature,
                    'stats': self.consistency_checker.get_stats()
                }
            except Exception as e:
                info['consistency_check'] = {
                    'error': str(e)
                }
        
        return info
    
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("EmbeddingManager 清理完成")
    
    def register_model_signature(self) -> None:
        """注册当前模型签名"""
        if not self.consistency_checker or not self.model:
            return
        
        try:
            # 创建当前模型签名
            current_signature = create_model_signature(
                model_name=self.model_name,
                model_type="sentence_transformer",
                dimension=self.embedding_dim,
                max_length=self.max_length,
                normalize=self.normalize_embeddings,
                metadata={
                    'device': str(self.device),
                    'batch_size': self.batch_size
                }
            )
            
            # 注册模型签名
            model_id = f"embedding_manager_{self.model_name}"
            self.consistency_checker.register_model(model_id, current_signature)
            
            logger.info(f"Model signature registered: {model_id}")
                    
        except Exception as e:
            logger.error(f"Error during model signature registration: {e}")
    
    def get_model_signature(self) -> Optional[Dict[str, Any]]:
        """获取当前模型签名"""
        if not self.model:
            return None
            
        try:
            signature = create_model_signature(
                model_name=self.model_name,
                model_type="sentence_transformer",
                dimension=self.embedding_dim,
                max_length=self.max_length,
                normalize=self.normalize_embeddings,
                metadata={
                    'device': str(self.device),
                    'batch_size': self.batch_size
                }
            )
            return signature.to_dict()
        except Exception as e:
            logger.error(f"Failed to create model signature: {e}")
            return None
    
    def validate_model_consistency(self, other_signature: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """验证模型一致性"""
        if not self.model:
            return True, None
        
        try:
            # 创建当前模型签名
            current_signature = self.get_model_signature()
            if not current_signature:
                return False, "Failed to create model signature"
            
            # 如果提供了其他签名，进行比较
            if other_signature:
                # 检查关键属性是否一致
                key_attrs = ['model_name', 'model_type', 'dimension', 'normalize']
                for attr in key_attrs:
                    if current_signature.get(attr) != other_signature.get(attr):
                        return False, f"Inconsistent {attr}: {current_signature.get(attr)} vs {other_signature.get(attr)}"
                
                return True, "Model signatures are consistent"
            
            # 如果有consistency_checker，使用它进行验证
            if self.consistency_checker:
                model_id = f"embedding_manager_{self.model_name}"
                violations = self.consistency_checker.get_violations()
                if violations:
                    return False, f"Found {len(violations)} consistency violations"
            
            return True, "Model consistency validated"
            
        except Exception as e:
            logger.error(f"Error during model consistency validation: {e}")
            return False, str(e)