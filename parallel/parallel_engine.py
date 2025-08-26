#!/usr/bin/env python3
"""
通用并行处理引擎

该模块提供了一个通用的并行处理框架，支持多种并行策略和任务类型。
可以被main.py、main_musique.py以及其他入口文件调用。

主要功能：
1. 多种并行策略（数据复制、数据分割、混合策略）
2. 灵活的任务分发机制
3. 智能结果汇总
4. 性能监控和统计
5. 错误处理和重试机制
"""

import os
import json
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from llm import LocalLLM
from llm.multi_model_client import MultiModelClient


class ParallelStrategy(Enum):
    """并行处理策略"""
    DATA_REPLICATION = "data_replication"  # 数据复制：多个模型处理相同数据
    DATA_SPLITTING = "data_splitting"      # 数据分割：多个模型处理不同数据
    TASK_DISTRIBUTION = "task_distribution" # 任务分发：不同模型处理不同类型任务
    HYBRID = "hybrid"                      # 混合策略：根据数据量和任务类型自动选择


class ProcessingMode(Enum):
    """处理模式"""
    MULTI_MODEL_CLIENT = "multi_model_client"  # 使用MultiModelClient
    SEPARATE_INSTANCES = "separate_instances"   # 使用独立的LLM实例
    AUTO = "auto"                              # 自动选择最优模式


@dataclass
class ParallelTask:
    """并行任务定义"""
    task_id: str
    data: Any
    task_type: str = "default"
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelResult:
    """并行处理结果"""
    task_id: str
    results: List[Dict[str, Any]]  # 各个模型的结果
    aggregated_result: Dict[str, Any]  # 聚合后的结果
    processing_time: float
    strategy_used: ParallelStrategy
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelStats:
    """并行处理统计信息"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    throughput: float = 0.0  # 任务/秒
    
    def update(self, result: ParallelResult):
        """更新统计信息"""
        self.total_tasks += 1
        if result.success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.total_processing_time += result.processing_time
        self.avg_processing_time = self.total_processing_time / self.total_tasks
        
        strategy_name = result.strategy_used.value
        self.strategy_usage[strategy_name] = self.strategy_usage.get(strategy_name, 0) + 1
        
        if self.total_processing_time > 0:
            self.throughput = self.total_tasks / self.total_processing_time


class TaskProcessor(ABC):
    """任务处理器抽象基类"""
    
    @abstractmethod
    def process_single_task(self, task: ParallelTask, llm: Union[LocalLLM, MultiModelClient], **kwargs) -> Dict[str, Any]:
        """处理单个任务"""
        pass
    
    @abstractmethod
    def aggregate_results(self, results: List[Dict[str, Any]], task: ParallelTask) -> Dict[str, Any]:
        """聚合多个结果"""
        pass
    
    def validate_task(self, task: ParallelTask) -> bool:
        """验证任务是否有效"""
        return task.task_id is not None and task.data is not None
    
    def preprocess_task(self, task: ParallelTask) -> ParallelTask:
        """任务预处理"""
        return task
    
    def postprocess_result(self, result: Dict[str, Any], task: ParallelTask) -> Dict[str, Any]:
        """结果后处理"""
        return result


class ParallelEngine:
    """通用并行处理引擎"""
    
    def __init__(self,
                 max_workers: int = 4,
                 processing_mode: ProcessingMode = ProcessingMode.AUTO,
                 default_strategy: ParallelStrategy = ParallelStrategy.HYBRID,
                 enable_stats: bool = True,
                 debug: bool = False):
        
        self.max_workers = max_workers
        self.processing_mode = processing_mode
        self.default_strategy = default_strategy
        self.enable_stats = enable_stats
        self.debug = debug
        
        # 统计信息
        self.stats = ParallelStats() if enable_stats else None
        self._lock = threading.Lock()
        
        # 初始化模型客户端
        self._init_model_clients()
        
        logger.info(f"ParallelEngine initialized with {max_workers} workers, mode: {processing_mode.value}")
    
    def _init_model_clients(self):
        """初始化模型客户端"""
        if self.processing_mode == ProcessingMode.MULTI_MODEL_CLIENT:
            logger.info("Initializing MultiModelClient for parallel processing...")
            self.multi_model_client = MultiModelClient()
            self.separate_llms = None
        elif self.processing_mode == ProcessingMode.SEPARATE_INSTANCES:
            logger.info("Initializing separate LocalLLM instances...")
            self.multi_model_client = None
            self.separate_llms = [LocalLLM() for _ in range(min(self.max_workers, 4))]  # 最多4个实例
        else:  # AUTO
            logger.info("Auto mode: initializing both MultiModelClient and separate instances...")
            try:
                self.multi_model_client = MultiModelClient()
                self.separate_llms = [LocalLLM() for _ in range(2)]  # 备用实例
                logger.info("Auto mode: MultiModelClient available, will prefer it")
            except Exception as e:
                logger.warning(f"MultiModelClient initialization failed: {e}, falling back to separate instances")
                self.multi_model_client = None
                self.separate_llms = [LocalLLM() for _ in range(min(self.max_workers, 4))]
    
    def _select_processing_mode(self, task_count: int) -> ProcessingMode:
        """根据任务数量选择最优处理模式"""
        if self.processing_mode != ProcessingMode.AUTO:
            return self.processing_mode
        
        # 自动选择逻辑
        if self.multi_model_client is not None:
            if task_count >= 4:  # 大批量任务优先使用MultiModelClient
                return ProcessingMode.MULTI_MODEL_CLIENT
            else:
                return ProcessingMode.SEPARATE_INSTANCES  # 小批量任务使用独立实例
        else:
            return ProcessingMode.SEPARATE_INSTANCES
    
    def _select_strategy(self, tasks: List[ParallelTask]) -> ParallelStrategy:
        """根据任务特征选择并行策略"""
        if self.default_strategy != ParallelStrategy.HYBRID:
            return self.default_strategy
        
        # 混合策略的自动选择逻辑
        task_count = len(tasks)
        
        # 检查任务类型的多样性
        task_types = set(task.task_type for task in tasks)
        
        if len(task_types) > 1:
            # 多种任务类型，使用任务分发策略
            return ParallelStrategy.TASK_DISTRIBUTION
        elif task_count >= 6:
            # 大量同类任务，使用数据分割策略
            return ParallelStrategy.DATA_SPLITTING
        else:
            # 少量任务，使用数据复制策略提高可靠性
            return ParallelStrategy.DATA_REPLICATION
    
    def process_tasks(self, 
                     tasks: List[ParallelTask], 
                     processor: TaskProcessor,
                     strategy: Optional[ParallelStrategy] = None) -> List[ParallelResult]:
        """批量处理任务"""
        
        if not tasks:
            logger.warning("No tasks to process")
            return []
        
        # 选择处理策略
        selected_strategy = strategy or self._select_strategy(tasks)
        selected_mode = self._select_processing_mode(len(tasks))
        
        logger.info(f"Processing {len(tasks)} tasks with strategy: {selected_strategy.value}, mode: {selected_mode.value}")
        
        # 验证和预处理任务
        valid_tasks = []
        for task in tasks:
            if processor.validate_task(task):
                valid_tasks.append(processor.preprocess_task(task))
            else:
                logger.warning(f"Invalid task skipped: {task.task_id}")
        
        if not valid_tasks:
            logger.error("No valid tasks to process")
            return []
        
        # 根据策略处理任务
        if selected_strategy == ParallelStrategy.DATA_REPLICATION:
            return self._process_with_replication(valid_tasks, processor, selected_mode)
        elif selected_strategy == ParallelStrategy.DATA_SPLITTING:
            return self._process_with_splitting(valid_tasks, processor, selected_mode)
        elif selected_strategy == ParallelStrategy.TASK_DISTRIBUTION:
            return self._process_with_distribution(valid_tasks, processor, selected_mode)
        else:  # HYBRID - 应该不会到这里，因为上面已经选择了具体策略
            return self._process_with_replication(valid_tasks, processor, selected_mode)
    
    def _process_with_replication(self, 
                                 tasks: List[ParallelTask], 
                                 processor: TaskProcessor,
                                 mode: ProcessingMode) -> List[ParallelResult]:
        """数据复制策略处理"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for task in tasks:
                if mode == ProcessingMode.MULTI_MODEL_CLIENT:
                    future = executor.submit(self._process_task_with_multi_model, task, processor)
                else:
                    future = executor.submit(self._process_task_with_replication, task, processor)
                futures.append((future, task))
            
            for future, task in futures:
                try:
                    result = future.result()
                    if self.enable_stats:
                        self.stats.update(result)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    error_result = ParallelResult(
                        task_id=task.task_id,
                        results=[],
                        aggregated_result={},
                        processing_time=0.0,
                        strategy_used=ParallelStrategy.DATA_REPLICATION,
                        success=False,
                        error_message=str(e)
                    )
                    if self.enable_stats:
                        self.stats.update(error_result)
                    results.append(error_result)
        
        return results
    
    def _process_with_splitting(self, 
                               tasks: List[ParallelTask], 
                               processor: TaskProcessor,
                               mode: ProcessingMode) -> List[ParallelResult]:
        """数据分割策略处理"""
        results = []
        
        # 将任务分组
        task_groups = self._split_tasks(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i, task_group in enumerate(task_groups):
                if mode == ProcessingMode.MULTI_MODEL_CLIENT:
                    future = executor.submit(self._process_task_group_with_multi_model, task_group, processor, i)
                else:
                    llm_index = i % len(self.separate_llms) if self.separate_llms else 0
                    future = executor.submit(self._process_task_group, task_group, processor, llm_index)
                futures.append(future)
            
            for future in futures:
                try:
                    group_results = future.result()
                    for result in group_results:
                        if self.enable_stats:
                            self.stats.update(result)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Task group processing failed: {e}")
        
        return results
    
    def _process_with_distribution(self, 
                                  tasks: List[ParallelTask], 
                                  processor: TaskProcessor,
                                  mode: ProcessingMode) -> List[ParallelResult]:
        """任务分发策略处理"""
        results = []
        
        # 按任务类型分组
        task_groups = self._group_tasks_by_type(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for task_type, task_group in task_groups.items():
                if mode == ProcessingMode.MULTI_MODEL_CLIENT:
                    future = executor.submit(self._process_typed_task_group_with_multi_model, task_group, processor, task_type)
                else:
                    # 为不同类型的任务分配不同的LLM实例
                    llm_index = hash(task_type) % len(self.separate_llms) if self.separate_llms else 0
                    future = executor.submit(self._process_typed_task_group, task_group, processor, llm_index, task_type)
                futures.append(future)
            
            for future in futures:
                try:
                    group_results = future.result()
                    for result in group_results:
                        if self.enable_stats:
                            self.stats.update(result)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Typed task group processing failed: {e}")
        
        return results
    
    def _process_task_with_multi_model(self, task: ParallelTask, processor: TaskProcessor) -> ParallelResult:
        """使用MultiModelClient处理单个任务"""
        start_time = time.time()
        
        try:
            # 使用MultiModelClient的并行能力
            result = processor.process_single_task(task, self.multi_model_client)
            result = processor.postprocess_result(result, task)
            
            return ParallelResult(
                task_id=task.task_id,
                results=[result],
                aggregated_result=result,
                processing_time=time.time() - start_time,
                strategy_used=ParallelStrategy.DATA_REPLICATION,
                success=True
            )
        except Exception as e:
            logger.error(f"MultiModelClient processing failed for task {task.task_id}: {e}")
            return ParallelResult(
                task_id=task.task_id,
                results=[],
                aggregated_result={},
                processing_time=time.time() - start_time,
                strategy_used=ParallelStrategy.DATA_REPLICATION,
                success=False,
                error_message=str(e)
            )
    
    def _process_task_with_replication(self, task: ParallelTask, processor: TaskProcessor) -> ParallelResult:
        """使用数据复制策略处理单个任务"""
        start_time = time.time()
        
        try:
            # 使用多个LLM实例并行处理相同任务
            with ThreadPoolExecutor(max_workers=min(len(self.separate_llms), 2)) as executor:
                futures = []
                for i, llm in enumerate(self.separate_llms[:2]):  # 最多使用2个实例
                    future = executor.submit(processor.process_single_task, task, llm)
                    futures.append(future)
                
                results = []
                for future in futures:
                    try:
                        result = future.result()
                        result = processor.postprocess_result(result, task)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"One instance failed for task {task.task_id}: {e}")
            
            if not results:
                raise Exception("All instances failed")
            
            # 聚合结果
            aggregated_result = processor.aggregate_results(results, task)
            
            return ParallelResult(
                task_id=task.task_id,
                results=results,
                aggregated_result=aggregated_result,
                processing_time=time.time() - start_time,
                strategy_used=ParallelStrategy.DATA_REPLICATION,
                success=True
            )
        except Exception as e:
            logger.error(f"Replication processing failed for task {task.task_id}: {e}")
            return ParallelResult(
                task_id=task.task_id,
                results=[],
                aggregated_result={},
                processing_time=time.time() - start_time,
                strategy_used=ParallelStrategy.DATA_REPLICATION,
                success=False,
                error_message=str(e)
            )
    
    def _split_tasks(self, tasks: List[ParallelTask]) -> List[List[ParallelTask]]:
        """将任务分割成多个组"""
        group_size = max(1, len(tasks) // self.max_workers)
        groups = []
        
        for i in range(0, len(tasks), group_size):
            groups.append(tasks[i:i + group_size])
        
        return groups
    
    def _group_tasks_by_type(self, tasks: List[ParallelTask]) -> Dict[str, List[ParallelTask]]:
        """按任务类型分组"""
        groups = {}
        for task in tasks:
            task_type = task.task_type
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(task)
        return groups
    
    def _process_task_group(self, task_group: List[ParallelTask], processor: TaskProcessor, llm_index: int) -> List[ParallelResult]:
        """处理任务组"""
        results = []
        llm = self.separate_llms[llm_index] if self.separate_llms else LocalLLM()
        
        for task in task_group:
            start_time = time.time()
            try:
                result = processor.process_single_task(task, llm)
                result = processor.postprocess_result(result, task)
                
                parallel_result = ParallelResult(
                    task_id=task.task_id,
                    results=[result],
                    aggregated_result=result,
                    processing_time=time.time() - start_time,
                    strategy_used=ParallelStrategy.DATA_SPLITTING,
                    success=True
                )
                results.append(parallel_result)
            except Exception as e:
                logger.error(f"Task {task.task_id} failed in group processing: {e}")
                error_result = ParallelResult(
                    task_id=task.task_id,
                    results=[],
                    aggregated_result={},
                    processing_time=time.time() - start_time,
                    strategy_used=ParallelStrategy.DATA_SPLITTING,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def _process_task_group_with_multi_model(self, task_group: List[ParallelTask], processor: TaskProcessor, group_index: int) -> List[ParallelResult]:
        """使用MultiModelClient处理任务组"""
        results = []
        
        for task in task_group:
            result = self._process_task_with_multi_model(task, processor)
            results.append(result)
        
        return results
    
    def _process_typed_task_group(self, task_group: List[ParallelTask], processor: TaskProcessor, llm_index: int, task_type: str) -> List[ParallelResult]:
        """处理特定类型的任务组"""
        results = []
        llm = self.separate_llms[llm_index] if self.separate_llms else LocalLLM()
        
        logger.info(f"Processing {len(task_group)} tasks of type '{task_type}' with LLM instance {llm_index}")
        
        for task in task_group:
            start_time = time.time()
            try:
                result = processor.process_single_task(task, llm)
                result = processor.postprocess_result(result, task)
                
                parallel_result = ParallelResult(
                    task_id=task.task_id,
                    results=[result],
                    aggregated_result=result,
                    processing_time=time.time() - start_time,
                    strategy_used=ParallelStrategy.TASK_DISTRIBUTION,
                    success=True,
                    metadata={'task_type': task_type, 'llm_index': llm_index}
                )
                results.append(parallel_result)
            except Exception as e:
                logger.error(f"Task {task.task_id} of type '{task_type}' failed: {e}")
                error_result = ParallelResult(
                    task_id=task.task_id,
                    results=[],
                    aggregated_result={},
                    processing_time=time.time() - start_time,
                    strategy_used=ParallelStrategy.TASK_DISTRIBUTION,
                    success=False,
                    error_message=str(e),
                    metadata={'task_type': task_type, 'llm_index': llm_index}
                )
                results.append(error_result)
        
        return results
    
    def _process_typed_task_group_with_multi_model(self, task_group: List[ParallelTask], processor: TaskProcessor, task_type: str) -> List[ParallelResult]:
        """使用MultiModelClient处理特定类型的任务组"""
        results = []
        
        logger.info(f"Processing {len(task_group)} tasks of type '{task_type}' with MultiModelClient")
        
        for task in task_group:
            result = self._process_task_with_multi_model(task, processor)
            result.strategy_used = ParallelStrategy.TASK_DISTRIBUTION
            result.metadata['task_type'] = task_type
            results.append(result)
        
        return results
    
    def get_stats(self) -> Optional[ParallelStats]:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        if self.enable_stats:
            self.stats = ParallelStats()
    
    def cleanup(self):
        """清理资源"""
        if self.multi_model_client:
            try:
                self.multi_model_client.cleanup()
            except:
                pass
        
        if self.separate_llms:
            for llm in self.separate_llms:
                try:
                    llm.cleanup()
                except:
                    pass
        
        logger.info("ParallelEngine cleanup completed")