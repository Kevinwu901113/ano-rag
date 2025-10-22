"""
vLLM 原子笔记生成器
专门用于vLLM双实例的原子笔记生成，支持长度分桶调度和批处理优化
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config

from .atomic_note_generator import AtomicNoteGenerator
from .vllm_openai_client import VllmOpenAIClient
from .bucket_scheduler import BucketScheduler, BatchedBucketScheduler
from .factory import LLMFactory
from utils.json_utils import extract_json_from_response
from utils.notes_parser import parse_notes_response, filter_valid_notes


class VllmAtomicNoteGenerator(AtomicNoteGenerator):
    """
    vLLM 原子笔记生成器
    
    继承自AtomicNoteGenerator，专门针对vLLM双实例优化：
    - 使用长度分桶调度器提升批处理效率
    - 支持异步并发处理
    - 保持与现有接口的兼容性
    """
    
    def __init__(self, llm=None, max_workers: Optional[int] = None, **kwargs):
        """
        初始化vLLM原子笔记生成器
        
        Args:
            llm: LocalLLM实例（可选，会被vLLM客户端替代）
            max_workers: 最大工作线程数
            **kwargs: 其他配置参数
        """
        # 调用父类初始化（但不会实际使用传入的llm）
        super().__init__(llm, max_workers)
        
        # 初始化vLLM客户端
        self.vllm_client = self._init_vllm_client(**kwargs)
        
        # 初始化分桶调度器
        bucket_edges = config.get('llm.note_generator.bucket_edges', [64, 128, 256, 512, 1024])
        self.bucket_scheduler = BatchedBucketScheduler(
            bucket_edges=bucket_edges,
            batch_size=32
        )
        
        # 性能统计
        self.vllm_stats = {
            'total_requests': 0,
            'bucket_distribution': {},
            'processing_time': 0,
            'throughput_qps': 0
        }
        
        logger.info("VllmAtomicNoteGenerator initialized with bucket scheduler")
    
    def _init_vllm_client(self, **kwargs) -> VllmOpenAIClient:
        """初始化vLLM客户端"""
        try:
            # 从配置或参数获取vLLM设置
            endpoints = kwargs.get('endpoints') or config.get('llm.note_generator.endpoints', [])
            model = kwargs.get('model') or config.get('llm.note_generator.model', 'Qwen/Qwen2.5-7B-Instruct')
            
            if not endpoints:
                raise ValueError("vLLM endpoints not configured")
            
            # 使用工厂创建vLLM客户端
            client = LLMFactory.create_provider('vllm-openai', **kwargs)
            logger.info(f"vLLM client initialized with {len(endpoints)} endpoints")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise
    
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        生成原子笔记（主入口方法）
        
        Args:
            text_chunks: 文本块列表
            progress_tracker: 进度跟踪器
            
        Returns:
            原子笔记列表
        """
        logger.info(f"Generating atomic notes for {len(text_chunks)} chunks using vLLM")
        
        # 重置统计信息
        self.reset_processing_stats()
        start_time = time.time()
        
        try:
            # 使用异步处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                atomic_notes = loop.run_until_complete(
                    self._generate_atomic_notes_async(text_chunks, progress_tracker)
                )
            finally:
                loop.close()
            
            # 计算处理时间和吞吐量
            processing_time = time.time() - start_time
            self.vllm_stats['processing_time'] = processing_time
            self.vllm_stats['throughput_qps'] = len(text_chunks) / processing_time if processing_time > 0 else 0
            
            # 记录统计信息
            logger.info(f"vLLM processing completed: {len(atomic_notes)} notes generated in {processing_time:.2f}s "
                       f"(QPS: {self.vllm_stats['throughput_qps']:.2f})")
            
            return atomic_notes
            
        except Exception as e:
            logger.error(f"vLLM atomic note generation failed: {e}")
            # 回退到父类方法
            logger.info("Falling back to sequential processing")
            return super().generate_atomic_notes(text_chunks, progress_tracker)
    
    async def _generate_atomic_notes_async(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        异步生成原子笔记
        
        Args:
            text_chunks: 文本块列表
            progress_tracker: 进度跟踪器
            
        Returns:
            原子笔记列表
        """
        if not text_chunks:
            return []
        
        # 获取系统提示
        system_prompt = self._get_atomic_note_system_prompt()
        
        # 将请求添加到分桶调度器
        request_ids = []
        chunk_mapping = {}  # request_id -> chunk_data 映射
        
        for chunk_data in text_chunks:
            prompt = self._format_atomic_note_prompt(chunk_data)
            request_id = self.bucket_scheduler.add_request(prompt, system_prompt)
            request_ids.append(request_id)
            chunk_mapping[request_id] = chunk_data
        
        # 记录分桶统计
        bucket_stats = self.bucket_scheduler.get_bucket_stats()
        self.vllm_stats['bucket_distribution'] = bucket_stats
        logger.info(f"Requests distributed across buckets: {bucket_stats}")
        
        # 使用分桶调度器处理请求
        results = await self.bucket_scheduler.process_with_client(self.vllm_client, system_prompt)
        
        # 处理结果并转换为原子笔记格式
        atomic_notes = []
        for request_id in request_ids:
            chunk_data = chunk_mapping[request_id]
            response = results.get(request_id, "")
            
            if response:
                try:
                    notes = self._process_llm_response(response, chunk_data)
                    atomic_notes.extend(notes)
                except Exception as e:
                    logger.error(f"Failed to process response for request {request_id}: {e}")
                    # 创建回退笔记
                    fallback_note = self._create_fallback_note(chunk_data)
                    if fallback_note:
                        atomic_notes.append(fallback_note)
            else:
                logger.warning(f"Empty response for request {request_id}")
                # 创建回退笔记
                fallback_note = self._create_fallback_note(chunk_data)
                if fallback_note:
                    atomic_notes.append(fallback_note)
        
        # 更新统计信息
        self.vllm_stats['total_requests'] = len(request_ids)
        self.processing_stats['total_chunks_processed'] = len(text_chunks)
        self.processing_stats['total_notes_generated'] = len(atomic_notes)
        
        return atomic_notes
    
    def _process_llm_response(self, response: str, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理LLM响应并转换为原子笔记格式
        
        Args:
            response: LLM响应文本
            chunk_data: 原始文本块数据
            
        Returns:
            原子笔记列表
        """
        try:
            # 解析响应
            notes_data = parse_notes_response(response, sentinel=self.sentinel_char)
            
            if not notes_data:
                logger.debug(f"No valid notes parsed from response: {response[:200]}...")
                return []
            
            # 转换为原子笔记格式
            atomic_notes = []
            for note_data in notes_data:
                try:
                    atomic_note = self._convert_to_atomic_note_format(note_data, chunk_data)
                    if atomic_note:
                        atomic_notes.append(atomic_note)
                except Exception as e:
                    logger.error(f"Failed to convert note to atomic format: {e}")
                    continue
            
            # 过滤有效笔记
            valid_notes = filter_valid_notes(atomic_notes)
            
            return valid_notes
            
        except Exception as e:
            logger.error(f"Failed to process LLM response: {e}")
            return []
    
    def _format_atomic_note_prompt(self, chunk_data: Dict[str, Any]) -> str:
        """
        格式化原子笔记生成提示
        
        Args:
            chunk_data: 文本块数据
            
        Returns:
            格式化的提示文本
        """
        text = chunk_data.get('text', '')
        
        # 使用父类的提示格式化方法
        _, user_prompt_template = self._get_atomic_note_prompts()
        
        # 格式化提示
        prompt = user_prompt_template.format(text=text)
        
        return prompt
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        try:
            vllm_health = await self.vllm_client.health_check()
            
            return {
                "status": "healthy",
                "vllm_endpoints": vllm_health,
                "bucket_scheduler": {
                    "bucket_count": len(self.bucket_scheduler.buckets),
                    "bucket_stats": self.bucket_scheduler.get_bucket_stats()
                },
                "processing_stats": self.vllm_stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计数据
        """
        return {
            "vllm_stats": self.vllm_stats,
            "processing_stats": self.processing_stats,
            "bucket_distribution": self.vllm_stats.get('bucket_distribution', {})
        }
    
    async def close(self):
        """关闭资源"""
        try:
            await self.vllm_client.close()
            logger.info("VllmAtomicNoteGenerator closed")
        except Exception as e:
            logger.error(f"Error closing VllmAtomicNoteGenerator: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            # 尝试清理资源
            if hasattr(self, 'vllm_client'):
                # 如果有事件循环，尝试关闭
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.close())
                except Exception:
                    pass
        except Exception:
            pass


def create_vllm_note_generator(**kwargs) -> VllmAtomicNoteGenerator:
    """
    创建vLLM原子笔记生成器的工厂函数
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        VllmAtomicNoteGenerator实例
    """
    try:
        return VllmAtomicNoteGenerator(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create vLLM note generator: {e}")
        raise