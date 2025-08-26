#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载辅助函数

提供统一的外部配置文件加载功能，支持相对路径和绝对路径。
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_external_config(config_file_path: str, 
                        base_config: Optional[Dict[str, Any]] = None,
                        config_key: Optional[str] = None) -> Dict[str, Any]:
    """
    从外部配置文件加载配置
    
    Args:
        config_file_path: 配置文件路径（支持相对路径和绝对路径）
        base_config: 基础配置字典，用于合并
        config_key: 配置文件中的特定键，如果指定则只返回该键的值
        
    Returns:
        加载的配置字典
    """
    if not config_file_path:
        logger.warning("Config file path is empty")
        return base_config or {}
    
    try:
        config_path = Path(config_file_path)
        
        # 如果是相对路径，相对于项目根目录
        if not config_path.is_absolute():
            # 项目根目录是当前文件的上两级目录
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        
        if not config_path.exists():
            logger.warning(f"External config file not found: {config_path}")
            return base_config or {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            external_config = yaml.safe_load(f)
        
        if external_config is None:
            logger.warning(f"External config file is empty: {config_path}")
            return base_config or {}
        
        # 如果指定了配置键，只返回该键的值
        if config_key:
            if config_key in external_config:
                result_config = external_config[config_key]
                logger.info(f"Loaded external config from {config_path} (key: {config_key})")
            else:
                logger.warning(f"Config key '{config_key}' not found in {config_path}")
                result_config = {}
        else:
            result_config = external_config
            logger.info(f"Loaded external config from {config_path}")
        
        # 如果有基础配置，进行合并
        if base_config:
            if isinstance(base_config, dict) and isinstance(result_config, dict):
                merged_config = base_config.copy()
                merged_config.update(result_config)
                return merged_config
            else:
                logger.warning("Cannot merge configs: both must be dictionaries")
                return result_config
        
        return result_config
        
    except Exception as e:
        logger.error(f"Failed to load external config from {config_file_path}: {e}")
        return base_config or {}

def merge_configs(base_config: Dict[str, Any], 
                 external_config: Dict[str, Any],
                 deep_merge: bool = True) -> Dict[str, Any]:
    """
    合并两个配置字典
    
    Args:
        base_config: 基础配置
        external_config: 外部配置
        deep_merge: 是否进行深度合并
        
    Returns:
        合并后的配置字典
    """
    if not isinstance(base_config, dict) or not isinstance(external_config, dict):
        logger.warning("Cannot merge configs: both must be dictionaries")
        return external_config if external_config else base_config
    
    result = base_config.copy()
    
    if deep_merge:
        for key, value in external_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value, deep_merge=True)
            else:
                result[key] = value
    else:
        result.update(external_config)
    
    return result

def load_config_with_fallback(config_file_path: str,
                             fallback_config: Dict[str, Any],
                             config_key: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件，如果失败则使用回退配置
    
    Args:
        config_file_path: 配置文件路径
        fallback_config: 回退配置
        config_key: 配置文件中的特定键
        
    Returns:
        加载的配置或回退配置
    """
    external_config = load_external_config(config_file_path, config_key=config_key)
    
    if external_config:
        return merge_configs(fallback_config, external_config)
    else:
        logger.info(f"Using fallback config for {config_file_path}")
        return fallback_config