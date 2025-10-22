"""
长度分桶调度器
按照文本长度将请求分桶，提升vLLM批处理效率
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from config import config


@dataclass
class BucketRequest:
    """分桶请求"""
    id: str
    prompt: str
    system_prompt: Optional[str]
    estimated_tokens: int
    bucket_id: int
    kwargs: Dict[str, Any]


class BucketScheduler:
    """
    长度分桶调度器
    
    将请求按照估计的token长度分桶，同桶内的请求并发处理，
    提升vLLM的连续批处理效率
    """
    
    def __init__(self, bucket_edges: List[int] = None, **kwargs):
        """
        初始化分桶调度器
        
        Args:
            bucket_edges: 分桶边界，例如 [64, 128, 256, 512, 1024]
            **kwargs: 其他配置参数
        """
        self.bucket_edges = bucket_edges or config.get('llm.note_generator.bucket_edges', [64, 128, 256, 512, 1024])
        self.buckets: Dict[int, List[BucketRequest]] = {}
        
        # 初始化桶
        for i in range(len(self.bucket_edges) + 1):
            self.buckets[i] = []
        
        logger.info(f"BucketScheduler initialized with edges: {self.bucket_edges}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        估计文本的token数量
        
        简单实现：使用字符数除以4作为粗略估计
        后续可以接入真实的tokenizer
        
        Args:
            text: 输入文本
            
        Returns:
            估计的token数量
        """
        if not text:
            return 0
        
        # 粗略估计：中文字符按1个token，英文单词按0.75个token计算
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        # 估算token数
        estimated_tokens = chinese_chars + (other_chars / 4)
        return int(estimated_tokens)
    
    def get_bucket_id(self, estimated_tokens: int) -> int:
        """
        根据token数量获取桶ID
        
        Args:
            estimated_tokens: 估计的token数量
            
        Returns:
            桶ID
        """
        for i, edge in enumerate(self.bucket_edges):
            if estimated_tokens <= edge:
                return i
        
        # 超过最大边界的放入最后一个桶
        return len(self.bucket_edges)
    
    def add_request(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        添加请求到分桶调度器
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            请求ID
        """
        # 生成请求ID
        request_id = hashlib.md5(f"{prompt}_{system_prompt}_{id(self)}".encode()).hexdigest()[:12]
        
        # 估计token数量
        full_text = f"{system_prompt or ''}\n{prompt}"
        estimated_tokens = self.estimate_tokens(full_text)
        
        # 获取桶ID
        bucket_id = self.get_bucket_id(estimated_tokens)
        
        # 创建请求对象
        request = BucketRequest(
            id=request_id,
            prompt=prompt,
            system_prompt=system_prompt,
            estimated_tokens=estimated_tokens,
            bucket_id=bucket_id,
            kwargs=kwargs
        )
        
        # 添加到对应桶
        self.buckets[bucket_id].append(request)
        
        logger.debug(f"Request {request_id} added to bucket {bucket_id} "
                    f"(estimated tokens: {estimated_tokens})")
        
        return request_id
    
    def get_bucket_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        获取桶统计信息
        
        Returns:
            桶统计信息
        """
        stats = {}
        
        for bucket_id, requests in self.buckets.items():
            if bucket_id < len(self.bucket_edges):
                bucket_range = f"≤{self.bucket_edges[bucket_id]}"
            else:
                bucket_range = f">{self.bucket_edges[-1]}"
            
            stats[bucket_id] = {
                "range": bucket_range,
                "count": len(requests),
                "avg_tokens": sum(req.estimated_tokens for req in requests) / len(requests) if requests else 0
            }
        
        return stats
    
    def get_batches_for_processing(self, min_batch_size: int = 1) -> List[Tuple[int, List[BucketRequest]]]:
        """
        获取用于处理的批次
        
        Args:
            min_batch_size: 最小批次大小
            
        Returns:
            (桶ID, 请求列表) 的列表
        """
        batches = []
        
        for bucket_id, requests in self.buckets.items():
            if len(requests) >= min_batch_size:
                # 复制请求列表并清空桶
                batch_requests = requests.copy()
                self.buckets[bucket_id] = []
                
                batches.append((bucket_id, batch_requests))
        
        return batches
    
    def clear_all_buckets(self) -> Dict[int, List[BucketRequest]]:
        """
        清空所有桶并返回所有请求
        
        Returns:
            所有桶的请求
        """
        all_requests = {}
        
        for bucket_id, requests in self.buckets.items():
            if requests:
                all_requests[bucket_id] = requests.copy()
                self.buckets[bucket_id] = []
        
        return all_requests
    
    async def process_with_client(self, client, system_prompt: str = None) -> Dict[str, str]:
        """
        使用指定客户端处理所有分桶请求
        
        Args:
            client: LLM客户端
            system_prompt: 统一的系统提示（可选）
            
        Returns:
            请求ID到结果的映射
        """
        results = {}
        
        # 获取所有批次
        batches = self.get_batches_for_processing(min_batch_size=1)
        
        if not batches:
            logger.info("No requests to process")
            return results
        
        # 记录统计信息
        total_requests = sum(len(requests) for _, requests in batches)
        logger.info(f"Processing {total_requests} requests in {len(batches)} buckets")
        
        # 并发处理各个桶
        tasks = []
        for bucket_id, requests in batches:
            task = self._process_bucket(client, bucket_id, requests, system_prompt)
            tasks.append(task)
        
        # 等待所有桶处理完成
        bucket_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        for i, bucket_result in enumerate(bucket_results):
            if isinstance(bucket_result, Exception):
                bucket_id = batches[i][0]
                logger.error(f"Bucket {bucket_id} processing failed: {bucket_result}")
            else:
                results.update(bucket_result)
        
        logger.info(f"Completed processing {len(results)} requests")
        return results
    
    async def _process_bucket(self, client, bucket_id: int, requests: List[BucketRequest], 
                            unified_system_prompt: str = None) -> Dict[str, str]:
        """
        处理单个桶的请求
        
        Args:
            client: LLM客户端
            bucket_id: 桶ID
            requests: 请求列表
            unified_system_prompt: 统一系统提示
            
        Returns:
            请求ID到结果的映射
        """
        if not requests:
            return {}
        
        logger.debug(f"Processing bucket {bucket_id} with {len(requests)} requests")
        
        # 准备批量请求
        prompts = []
        request_ids = []
        
        for request in requests:
            prompts.append(request.prompt)
            request_ids.append(request.id)
        
        try:
            # 使用统一的系统提示或第一个请求的系统提示
            system_prompt = unified_system_prompt or (requests[0].system_prompt if requests else None)
            
            # 批量调用客户端
            if hasattr(client, 'chat_many'):
                # 异步批量调用
                responses = await client.chat_many(prompts, system_prompt)
            else:
                # 回退到单个调用
                tasks = []
                for prompt in prompts:
                    if hasattr(client, 'chat_one'):
                        task = client.chat_one(prompt, system_prompt)
                    else:
                        # 同步调用的异步包装
                        task = asyncio.get_event_loop().run_in_executor(
                            None, client.generate, prompt, system_prompt
                        )
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 构建结果映射
            results = {}
            for i, (request_id, response) in enumerate(zip(request_ids, responses)):
                if isinstance(response, Exception):
                    logger.error(f"Request {request_id} failed: {response}")
                    results[request_id] = ""  # 失败时返回空字符串
                else:
                    results[request_id] = response
            
            logger.debug(f"Bucket {bucket_id} completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Bucket {bucket_id} processing failed: {e}")
            # 返回空结果
            return {request_id: "" for request_id in request_ids}


class BatchedBucketScheduler(BucketScheduler):
    """
    批处理分桶调度器
    
    扩展基础分桶调度器，支持更高级的批处理策略
    """
    
    def __init__(self, bucket_edges: List[int] = None, batch_size: int = 32, **kwargs):
        """
        初始化批处理分桶调度器
        
        Args:
            bucket_edges: 分桶边界
            batch_size: 批处理大小
            **kwargs: 其他配置参数
        """
        super().__init__(bucket_edges, **kwargs)
        self.batch_size = batch_size
        
        logger.info(f"BatchedBucketScheduler initialized with batch_size: {batch_size}")
    
    def get_batches_for_processing(self, min_batch_size: int = None) -> List[Tuple[int, List[BucketRequest]]]:
        """
        获取用于处理的批次（支持批大小控制）
        
        Args:
            min_batch_size: 最小批次大小（忽略，使用配置的batch_size）
            
        Returns:
            (桶ID, 请求列表) 的列表
        """
        batches = []
        
        for bucket_id, requests in self.buckets.items():
            # 按批大小分割请求
            while len(requests) >= self.batch_size:
                batch_requests = requests[:self.batch_size]
                requests = requests[self.batch_size:]
                batches.append((bucket_id, batch_requests))
            
            # 更新桶中剩余的请求
            self.buckets[bucket_id] = requests
        
        return batches
    
    def force_flush_all(self) -> List[Tuple[int, List[BucketRequest]]]:
        """
        强制刷新所有桶（包括未满的批次）
        
        Returns:
            所有批次
        """
        batches = []
        
        for bucket_id, requests in self.buckets.items():
            if requests:
                batches.append((bucket_id, requests.copy()))
                self.buckets[bucket_id] = []
        
        return batches