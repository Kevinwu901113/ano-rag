import asyncio
import concurrent.futures
from typing import List, Callable, Any, Optional, Union
from tqdm import tqdm
from loguru import logger
from .gpu_utils import GPUUtils

class BatchProcessor:
    """批处理器，用于高效的批量处理"""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4, use_gpu: bool = True):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_gpu = use_gpu and GPUUtils.is_cuda_available()
        self.preprocess_func = None
        self.postprocess_func = None
        self.error_handler = None
        self.logger = logger
    
    def _monitor_gpu_memory(self, stage: str):
        """监控GPU内存使用情况"""
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    cached = torch.cuda.memory_reserved() / (1024**3)
                    self.logger.debug(f"GPU Memory at {stage}: Allocated={allocated:.2f}GB, Cached={cached:.2f}GB")
            except ImportError:
                pass
    
    def _cleanup_gpu_memory(self, stage: str):
        """清理GPU内存"""
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug(f"GPU memory cleaned at {stage}")
            except ImportError:
                pass
    
    def process_batch(self, batch_data: List[Any], process_func: Callable, **kwargs) -> List[Any]:
        """处理单个批次的数据"""
        try:
            # GPU内存监控
            if self.use_gpu:
                self._monitor_gpu_memory("batch_start")
            
            # 应用预处理
            if self.preprocess_func:
                batch_data = [self.preprocess_func(item) for item in batch_data]
            
            # 处理批次
            results = process_func(batch_data, **kwargs)
            
            # 应用后处理
            if self.postprocess_func:
                results = [self.postprocess_func(item) for item in results]
            
            # GPU内存清理
            if self.use_gpu:
                self._cleanup_gpu_memory("batch_end")
            
            return results
        except Exception as e:
            self.logger.error(f"批次处理失败: {e}")
            # 出错时也要清理GPU内存
            if self.use_gpu:
                self._cleanup_gpu_memory("batch_error")
            
            if self.error_handler:
                return self.error_handler(batch_data, e)
            else:
                raise
        
    def process_batches(self, 
                       data: List[Any], 
                       process_func: Callable,
                       desc: str = "Processing",
                       progress_tracker: Optional[Any] = None,
                       **kwargs) -> List[Any]:
        """同步批处理数据"""
        if not data:
            return []
        
        # 初始化内存监控
        if self.use_gpu:
            self._monitor_gpu_memory("process_start")
        
        # 自适应调整批次大小
        actual_batch_size = self.adaptive_batch_size(len(data))
        self.logger.info(f"使用批次大小: {actual_batch_size}")
        
        results = []
        
        # 如果提供了外部进度跟踪器，使用它；否则创建内部tqdm进度条
        use_external_tracker = progress_tracker is not None
        if use_external_tracker:
            pbar = None
        else:
            pbar = tqdm(total=len(data), desc=desc)
        
        try:
            for i in range(0, len(data), actual_batch_size):
                batch = data[i:i + actual_batch_size]
                
                try:
                    if self.use_gpu:
                        batch_result = GPUUtils.batch_process_gpu(
                            batch, len(batch), process_func, use_gpu=True
                        )
                    else:
                        batch_result = process_func(batch, **kwargs)
                    
                    results.extend(batch_result)
                    if use_external_tracker:
                        # 使用外部进度跟踪器更新进度
                        for _ in range(len(batch)):
                            progress_tracker.update(1)
                    else:
                        pbar.update(len(batch))
                    
                    # 定期清理内存
                    if self.use_gpu and (i // actual_batch_size + 1) % 5 == 0:
                        self._cleanup_gpu_memory(f"batch_{i//actual_batch_size+1}")
                        
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # 尝试逐个处理
                    for item in batch:
                        try:
                            # 检查是否是原子笔记生成器的方法
                            if hasattr(process_func, '__self__') and hasattr(process_func.__self__, '_generate_single_atomic_note'):
                                # 直接调用单个处理方法，传递系统提示词
                                system_prompt = kwargs.get('system_prompt', process_func.__self__._get_atomic_note_system_prompt())
                                item_result = process_func.__self__._generate_single_atomic_note(item, system_prompt)
                                if isinstance(item_result, list):
                                    results.extend(item_result)
                                elif item_result:
                                    results.append(item_result)
                            else:
                                # 对于其他类型的处理函数，直接调用单个item处理
                                item_result = process_func(item, **kwargs)
                                # 确保结果被正确添加
                                if isinstance(item_result, list):
                                    results.extend(item_result)
                                else:
                                    results.append(item_result)
                        except Exception as item_e:
                            logger.error(f"Item processing failed: {item_e}")
                            # 创建一个基本的fallback结果
                            if hasattr(process_func, '__self__') and hasattr(process_func.__self__, '_create_fallback_note'):
                                fallback_result = process_func.__self__._create_fallback_note(item)
                                results.append(fallback_result)
                            else:
                                # 如果没有fallback方法，创建一个基本的错误结果
                                error_result = {
                                    'error': True,
                                    'error_message': str(item_e),
                                    'original_data': item
                                }
                                results.append(error_result)
                        
                        # 更新进度
                        if use_external_tracker:
                            progress_tracker.update(1)
                        else:
                            pbar.update(1)
        
        finally:
            # 最终清理
            if self.use_gpu:
                self._cleanup_gpu_memory("process_end")
                self._monitor_gpu_memory("process_final")
            
            # 关闭内部进度条（如果使用的话）
            if not use_external_tracker and pbar:
                pbar.close()
            
        return results
    
    async def process_batches_async(self,
                                   data: List[Any],
                                   process_func: Callable,
                                   desc: str = "Processing",
                                   **kwargs) -> List[Any]:
        """异步批处理"""
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_batch(batch):
            async with semaphore:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if self.use_gpu:
                        result = await loop.run_in_executor(
                            executor, 
                            lambda: GPUUtils.batch_process_gpu(
                                batch, len(batch), process_func, use_gpu=True
                            )
                        )
                    else:
                        result = await loop.run_in_executor(
                            executor, process_func, batch, **kwargs
                        )
                    return result
        
        # 创建批次
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        # 并行处理批次
        with tqdm(total=len(data), desc=desc) as pbar:
            tasks = []
            for batch in batches:
                task = asyncio.create_task(process_batch(batch))
                tasks.append(task)
            
            for task in asyncio.as_completed(tasks):
                try:
                    batch_result = await task
                    results.extend(batch_result)
                    pbar.update(len(batch_result))
                except Exception as e:
                    logger.error(f"Async batch processing failed: {e}")
        
        return results
    
    def parallel_process(self,
                        data_list: List[List[Any]],
                        process_func: Callable,
                        desc: str = "Parallel Processing",
                        **kwargs) -> List[List[Any]]:
        """并行处理多个数据列表（用于多查询并行处理）"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_data = {
                executor.submit(self.process_batches, data, process_func, f"{desc} {i}", **kwargs): i
                for i, data in enumerate(data_list)
            }
            
            # 收集结果
            with tqdm(total=len(data_list), desc=desc) as pbar:
                for future in concurrent.futures.as_completed(future_to_data):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Parallel processing failed: {e}")
                        results.append([])
                        pbar.update(1)
        
        return results
    
    def adaptive_batch_size(self, data_size: int, base_batch_size: int = None) -> int:
        """根据数据大小和内存限制动态调整批次大小"""
        if base_batch_size is None:
            base_batch_size = self.batch_size
        
        # GPU内存估算（优化版本）
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    # 获取GPU内存信息
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_allocated = torch.cuda.memory_allocated()
                    gpu_memory_cached = torch.cuda.memory_reserved()
                    gpu_memory_free = gpu_memory - gpu_memory_cached
                    
                    # 更精确的内存使用率计算
                    memory_usage_ratio = gpu_memory_allocated / gpu_memory
                    
                    # 根据内存使用情况动态调整
                    if memory_usage_ratio > 0.8:  # 高内存使用
                        adjusted_batch_size = max(1, base_batch_size // 4)
                    elif memory_usage_ratio > 0.6:  # 中等内存使用
                        adjusted_batch_size = max(1, base_batch_size // 2)
                    else:  # 低内存使用
                        memory_factor = min(gpu_memory_free / (1024**3), 8.0)  # GB为单位
                        adjusted_batch_size = int(base_batch_size * min(memory_factor / 2.0, 1.5))
                    
                    # 清理GPU缓存以释放内存
                    if memory_usage_ratio > 0.7:
                        torch.cuda.empty_cache()
                    
                    return max(1, min(adjusted_batch_size, data_size))
            except ImportError:
                pass
        
        # CPU内存估算（优化版本）
        import psutil
        memory_info = psutil.virtual_memory()
        available_memory = memory_info.available / (1024**3)  # GB
        memory_usage_ratio = memory_info.percent / 100.0
        
        # 根据内存使用率动态调整
        if memory_usage_ratio > 0.85 or available_memory < 1:
            return max(1, base_batch_size // 8)
        elif memory_usage_ratio > 0.7 or available_memory < 2:
            return max(1, base_batch_size // 4)
        elif memory_usage_ratio > 0.5 or available_memory < 4:
            return max(1, base_batch_size // 2)
        else:
            return min(base_batch_size, data_size)
    
    def chunk_large_data(self, data: List[Any], max_chunk_size: int = 10000) -> List[List[Any]]:
        """将大数据集分块处理"""
        chunks = []
        for i in range(0, len(data), max_chunk_size):
            chunk = data[i:i + max_chunk_size]
            chunks.append(chunk)
        return chunks