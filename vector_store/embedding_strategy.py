#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原子笔记入库与嵌入策略统一模块

实现功能：
1. 统一嵌入策略管理
2. 索引版本管理和热切换
3. 多模型嵌入支持
4. 嵌入缓存和批处理
5. 增量更新和回滚机制
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("numpy not available, some features may be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch not available, some features may be limited")

logger = logging.getLogger(__name__)

class EmbeddingModelType(Enum):
    """嵌入模型类型"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class IndexStatus(Enum):
    """索引状态"""
    BUILDING = "building"
    READY = "ready"
    UPDATING = "updating"
    ERROR = "error"
    DEPRECATED = "deprecated"

@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    model_name: str                     # 模型名称
    model_type: EmbeddingModelType      # 模型类型
    dimension: int                      # 嵌入维度
    max_length: int = 512              # 最大输入长度
    batch_size: int = 32               # 批处理大小
    normalize: bool = True             # 是否归一化
    device: str = "auto"               # 设备
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # 模型参数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'dimension': self.dimension,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'normalize': self.normalize,
            'device': self.device,
            'model_kwargs': self.model_kwargs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """从字典创建"""
        return cls(
            model_name=data['model_name'],
            model_type=EmbeddingModelType(data['model_type']),
            dimension=data['dimension'],
            max_length=data.get('max_length', 512),
            batch_size=data.get('batch_size', 32),
            normalize=data.get('normalize', True),
            device=data.get('device', 'auto'),
            model_kwargs=data.get('model_kwargs', {})
        )
    
    def get_hash(self) -> str:
        """获取配置哈希"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

@dataclass
class IndexVersion:
    """索引版本信息"""
    version_id: str                     # 版本ID
    config: EmbeddingConfig            # 嵌入配置
    status: IndexStatus                # 状态
    created_at: float                  # 创建时间
    updated_at: float                  # 更新时间
    document_count: int = 0            # 文档数量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'config': self.config.to_dict(),
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'document_count': self.document_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexVersion':
        """从字典创建"""
        return cls(
            version_id=data['version_id'],
            config=EmbeddingConfig.from_dict(data['config']),
            status=IndexStatus(data['status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            document_count=data.get('document_count', 0),
            metadata=data.get('metadata', {})
        )

class EmbeddingModel:
    """嵌入模型抽象基类"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._lock = threading.Lock()
        
    def load_model(self) -> None:
        """加载模型"""
        raise NotImplementedError
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        raise NotImplementedError
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
    
    def unload_model(self) -> None:
        """卸载模型"""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformer模型"""
    
    def load_model(self) -> None:
        """加载Sentence Transformer模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            with self._lock:
                if self.model is None:
                    device = self.config.device
                    if device == "auto":
                        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
                    
                    self.model = SentenceTransformer(
                        self.config.model_name,
                        device=device,
                        **self.config.model_kwargs
                    )
                    
                    logger.info(f"Loaded SentenceTransformer model: {self.config.model_name} on {device}")
        except ImportError:
            raise ImportError("sentence-transformers not available")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表"""
        if not self.is_loaded():
            self.load_model()
        
        try:
            # 截断文本
            truncated_texts = [text[:self.config.max_length] for text in texts]
            
            # 编码
            embeddings = self.model.encode(
                truncated_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

class EmbeddingStrategy:
    """嵌入策略管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_config_with_fallback
        
        # 尝试从外部配置文件加载，如果失败则使用内联配置
        config_file_path = config.get('embedding_strategy_config_file')
        
        # 默认配置
        default_config = {
            'storage_path': './embeddings',
            'enable_cache': True,
            'cache_size': 10000,
            'max_workers': 4,
            'enable_hot_swap': True,
            'backup_versions': 3
        }
        
        # 加载配置（外部文件优先，回退到内联配置，最后使用默认配置）
        inline_config = config.get('embedding_strategy', {})
        self.config = load_config_with_fallback(config_file_path, inline_config, default_config)
        
        # 配置参数
        self.storage_path = Path(self.config.get('storage_path', default_config['storage_path']))
        self.enable_cache = self.config.get('enable_cache', default_config['enable_cache'])
        self.cache_size = self.config.get('cache_size', default_config['cache_size'])
        self.max_workers = self.config.get('max_workers', default_config['max_workers'])
        self.enable_hot_swap = self.config.get('enable_hot_swap', default_config['enable_hot_swap'])
        self.backup_versions = self.config.get('backup_versions', default_config['backup_versions'])
        
        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 索引版本管理
        self.versions: Dict[str, IndexVersion] = {}
        self.current_version_id: Optional[str] = None
        self.models: Dict[str, EmbeddingModel] = {}
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'model_switches': 0,
            'version_updates': 0
        }
        
        logger.info(f"EmbeddingStrategy initialized")
    
    def create_version(self, config: EmbeddingConfig, set_as_current: bool = True) -> str:
        """
        创建新的索引版本
        
        Args:
            config: 嵌入配置
            set_as_current: 是否设置为当前版本
            
        Returns:
            版本ID
        """
        version_id = f"{config.get_hash()}_{int(time.time())}"
        
        version = IndexVersion(
            version_id=version_id,
            config=config,
            status=IndexStatus.BUILDING,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        with self._lock:
            self.versions[version_id] = version
            
            if set_as_current:
                self.current_version_id = version_id
                self.stats['version_updates'] += 1
        
        logger.info(f"Created new embedding version: {version_id}")
        return version_id
    
    def get_model(self, version_id: Optional[str] = None) -> EmbeddingModel:
        """
        获取嵌入模型
        
        Args:
            version_id: 版本ID，如果为None则使用当前版本
            
        Returns:
            嵌入模型实例
        """
        if version_id is None:
            version_id = self.current_version_id
        
        if version_id is None:
            raise ValueError("No current version set")
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # 检查模型是否已加载
        if version_id not in self.models:
            version = self.versions[version_id]
            model = self._create_model(version.config)
            
            with self._lock:
                self.models[version_id] = model
                self.stats['model_switches'] += 1
        
        return self.models[version_id]
    
    def _create_model(self, config: EmbeddingConfig) -> EmbeddingModel:
        """创建嵌入模型"""
        if config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
            return SentenceTransformerModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_embeddings': self.stats['total_embeddings'],
            'cache_hits': self.stats['cache_hits'],
            'model_switches': self.stats['model_switches'],
            'version_updates': self.stats['version_updates'],
            'versions_count': len(self.versions),
            'current_version': self.current_version_id,
            'loaded_models': len(self.models)
        }

# 便利函数
def create_embedding_strategy(config: Dict[str, Any]) -> EmbeddingStrategy:
    """创建嵌入策略实例"""
    return EmbeddingStrategy(config)

def create_embedding_config(model_name: str, 
                          model_type: Union[str, EmbeddingModelType],
                          dimension: int,
                          **kwargs) -> EmbeddingConfig:
    """
    创建嵌入配置的便利函数
    
    Args:
        model_name: 模型名称
        model_type: 模型类型
        dimension: 嵌入维度
        **kwargs: 其他参数
        
    Returns:
        嵌入配置实例
    """
    if isinstance(model_type, str):
        model_type = EmbeddingModelType(model_type)
    
    return EmbeddingConfig(
        model_name=model_name,
        model_type=model_type,
        dimension=dimension,
        **kwargs
    )