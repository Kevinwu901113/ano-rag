#!/usr/bin/env python3
"""
并行处理模块

该模块提供了通用的并行处理能力，支持多种并行策略和处理模式。
可以被项目中的任何入口文件调用，实现高效的并行处理。

主要组件：
- ParallelEngine: 核心并行处理引擎
- TaskProcessor: 任务处理器抽象基类
- ParallelInterface: 统一的并行处理接口
- 各种数据类和枚举类型
"""

from .parallel_engine import (
    ParallelEngine,
    TaskProcessor,
    ParallelTask,
    ParallelResult,
    ParallelStats,
    ParallelStrategy,
    ProcessingMode
)

from .parallel_interface import (
    ParallelInterface,
    DocumentTaskProcessor,
    QueryTaskProcessor,
    MusiqueTaskProcessor,
    create_parallel_interface
)

__all__ = [
    # 核心引擎
    'ParallelEngine',
    'TaskProcessor',
    
    # 数据类型
    'ParallelTask',
    'ParallelResult', 
    'ParallelStats',
    
    # 枚举类型
    'ParallelStrategy',
    'ProcessingMode',
    
    # 接口和处理器
    'ParallelInterface',
    'DocumentTaskProcessor',
    'QueryTaskProcessor', 
    'MusiqueTaskProcessor',
    'create_parallel_interface'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'AnoRAG Team'
__description__ = 'Universal parallel processing module for AnoRAG'