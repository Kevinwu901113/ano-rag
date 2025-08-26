#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨模型一致性与归一化模块

实现功能：
1. 跨模型一致性检查
2. 模型兼容性验证
3. 跨模型投影层
4. 模型混用阻止机制
5. 归一化处理流程
"""

import logging
import json
import hashlib
import warnings
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("numpy not available, some features may be limited")

logger = logging.getLogger(__name__)

class ConsistencyLevel(Enum):
    """一致性级别"""
    STRICT = "strict"          # 严格模式：完全禁止混用
    MODERATE = "moderate"      # 适中模式：允许兼容模型混用
    PERMISSIVE = "permissive"  # 宽松模式：允许混用但发出警告
    DISABLED = "disabled"      # 禁用模式：不进行检查

class ModelCompatibility(Enum):
    """模型兼容性"""
    IDENTICAL = "identical"        # 完全相同
    COMPATIBLE = "compatible"      # 兼容
    INCOMPATIBLE = "incompatible"  # 不兼容
    UNKNOWN = "unknown"            # 未知

@dataclass
class ModelSignature:
    """模型签名"""
    model_name: str                    # 模型名称
    model_type: str                    # 模型类型
    dimension: int                     # 嵌入维度
    max_length: int                    # 最大长度
    normalize: bool                    # 是否归一化
    signature_hash: str = ""           # 签名哈希
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __post_init__(self):
        """计算签名哈希"""
        if not self.signature_hash:
            signature_data = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'dimension': self.dimension,
                'max_length': self.max_length,
                'normalize': self.normalize
            }
            signature_str = json.dumps(signature_data, sort_keys=True)
            self.signature_hash = hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    def is_compatible_with(self, other: 'ModelSignature') -> ModelCompatibility:
        """检查与另一个模型签名的兼容性"""
        if self.signature_hash == other.signature_hash:
            return ModelCompatibility.IDENTICAL
        
        # 检查关键属性
        if (self.model_type == other.model_type and 
            self.dimension == other.dimension and
            self.normalize == other.normalize):
            return ModelCompatibility.COMPATIBLE
        
        return ModelCompatibility.INCOMPATIBLE

@dataclass
class ConsistencyViolation:
    """一致性违规记录"""
    violation_type: str                # 违规类型
    source_signature: ModelSignature  # 源模型签名
    target_signature: ModelSignature  # 目标模型签名
    severity: str                      # 严重程度
    message: str                       # 违规消息
    timestamp: float                   # 时间戳
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文信息

class ModelConsistencyChecker:
    """模型一致性检查器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('model_consistency', {})
        
        # 配置参数
        self.consistency_level = ConsistencyLevel(
            self.config.get('consistency_level', 'moderate')
        )
        self.max_violations = self.config.get('max_violations', 100)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # 状态管理
        self.registered_models: Dict[str, ModelSignature] = {}
        self.violations: List[ConsistencyViolation] = []
        self.compatibility_cache: Dict[Tuple[str, str], ModelCompatibility] = {}
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'violations_detected': 0,
            'models_registered': 0
        }
        
        logger.info(f"ModelConsistencyChecker initialized with level: {self.consistency_level.value}")
    
    def register_model(self, model_id: str, signature: ModelSignature) -> None:
        """
        注册模型签名
        
        Args:
            model_id: 模型ID
            signature: 模型签名
        """
        with self._lock:
            self.registered_models[model_id] = signature
            self.stats['models_registered'] += 1
        
        logger.info(f"Registered model: {model_id} ({signature.model_name})")
    
    def check_consistency(self, model_id_1: str, model_id_2: str) -> ModelCompatibility:
        """
        检查两个模型的一致性
        
        Args:
            model_id_1: 第一个模型ID
            model_id_2: 第二个模型ID
            
        Returns:
            模型兼容性
        """
        self.stats['total_checks'] += 1
        
        # 检查缓存
        cache_key = (model_id_1, model_id_2)
        if self.enable_caching and cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        # 获取模型签名
        if model_id_1 not in self.registered_models:
            raise ValueError(f"Model {model_id_1} not registered")
        if model_id_2 not in self.registered_models:
            raise ValueError(f"Model {model_id_2} not registered")
        
        signature_1 = self.registered_models[model_id_1]
        signature_2 = self.registered_models[model_id_2]
        
        # 检查兼容性
        compatibility = signature_1.is_compatible_with(signature_2)
        
        # 缓存结果
        if self.enable_caching:
            self.compatibility_cache[cache_key] = compatibility
            # 对称缓存
            self.compatibility_cache[(model_id_2, model_id_1)] = compatibility
        
        # 处理不兼容情况
        if compatibility == ModelCompatibility.INCOMPATIBLE:
            self._handle_incompatibility(model_id_1, model_id_2, signature_1, signature_2)
        
        return compatibility
    
    def _handle_incompatibility(self, model_id_1: str, model_id_2: str,
                               signature_1: ModelSignature, 
                               signature_2: ModelSignature) -> None:
        """处理模型不兼容情况"""
        violation = ConsistencyViolation(
            violation_type="model_incompatibility",
            source_signature=signature_1,
            target_signature=signature_2,
            severity="high",
            message=f"Models {model_id_1} and {model_id_2} are incompatible",
            timestamp=time.time(),
            context={'model_id_1': model_id_1, 'model_id_2': model_id_2}
        )
        
        self.violations.append(violation)
        self.stats['violations_detected'] += 1
        
        # 根据一致性级别处理
        if self.consistency_level == ConsistencyLevel.STRICT:
            raise ValueError(violation.message)
        elif self.consistency_level == ConsistencyLevel.MODERATE:
            warnings.warn(violation.message)
        elif self.consistency_level == ConsistencyLevel.PERMISSIVE:
            warnings.warn(violation.message)
        # DISABLED级别不做任何处理
    
    def get_violations(self, severity: Optional[str] = None) -> List[ConsistencyViolation]:
        """获取违规记录"""
        if severity is None:
            return self.violations.copy()
        return [v for v in self.violations if v.severity == severity]
    
    def clear_violations(self) -> None:
        """清除违规记录"""
        with self._lock:
            self.violations.clear()
        logger.info("Cleared all violations")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_checks': self.stats['total_checks'],
            'violations_detected': self.stats['violations_detected'],
            'models_registered': self.stats['models_registered'],
            'registered_models': len(self.registered_models),
            'cached_compatibilities': len(self.compatibility_cache),
            'consistency_level': self.consistency_level.value
        }

# 便利函数
def create_model_consistency_checker(config: Dict[str, Any]) -> ModelConsistencyChecker:
    """创建模型一致性检查器"""
    return ModelConsistencyChecker(config)

def create_model_signature(model_name: str, model_type: str, 
                          dimension: int, **kwargs) -> ModelSignature:
    """
    创建模型签名的便利函数
    
    Args:
        model_name: 模型名称
        model_type: 模型类型
        dimension: 嵌入维度
        **kwargs: 其他参数
        
    Returns:
        模型签名实例
    """
    return ModelSignature(
        model_name=model_name,
        model_type=model_type,
        dimension=dimension,
        max_length=kwargs.get('max_length', 512),
        normalize=kwargs.get('normalize', True),
        metadata=kwargs.get('metadata', {})
    )

def check_models_compatibility(checker: ModelConsistencyChecker,
                             model_ids: List[str]) -> Dict[Tuple[str, str], ModelCompatibility]:
    """
    批量检查多个模型的兼容性
    
    Args:
        checker: 一致性检查器
        model_ids: 模型ID列表
        
    Returns:
        兼容性检查结果字典
    """
    results = {}
    
    for i, model_id_1 in enumerate(model_ids):
        for model_id_2 in model_ids[i+1:]:
            compatibility = checker.check_consistency(model_id_1, model_id_2)
            results[(model_id_1, model_id_2)] = compatibility
    
    return results