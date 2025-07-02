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
        
    def process_batches(self, 
                       data: List[Any], 
                       process_func: Callable,
                       desc: str = "Processing",
                       **kwargs) -> List[Any]:
        """同步批处理"""
        results = []
        
        with tqdm(total=len(data), desc=desc) as pbar:
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                
                try:
                    if self.use_gpu:
                        batch_result = GPUUtils.batch_process_gpu(
                            batch, len(batch), process_func, use_gpu=True
                        )
                    else:
                        batch_result = process_func(batch, **kwargs)
                    
                    results.extend(batch_result)
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # 尝试逐个处理
                    for item in batch:
                        try:
                            item_result = process_func([item], **kwargs)
                            results.extend(item_result)
                        except Exception as item_e:
                            logger.error(f"Item processing failed: {item_e}")
                        pbar.update(1)
        
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
    
    def adaptive_batch_size(self, data_size: int, memory_limit: Optional[int] = None) -> int:
        """根据数据大小和内存限制自适应调整批次大小"""
        if memory_limit is None:
            # 获取GPU内存信息
            memory_info = GPUUtils.get_memory_info()
            available_memory = memory_info['total'] - memory_info['allocated']
            memory_limit = int(available_memory * 0.8)  # 使用80%的可用内存
        
        # 估算每个样本的内存使用量（简化估算）
        estimated_memory_per_sample = 1024 * 1024  # 1MB per sample (rough estimate)
        
        # 计算最大批次大小
        max_batch_size = max(1, memory_limit // estimated_memory_per_sample)
        
        # 返回较小的值
        return min(self.batch_size, max_batch_size, data_size)
    
    def chunk_large_data(self, data: List[Any], max_chunk_size: int = 10000) -> List[List[Any]]:
        """将大数据集分块处理"""
        chunks = []
        for i in range(0, len(data), max_chunk_size):
            chunk = data[i:i + max_chunk_size]
            chunks.append(chunk)
        return chunks