"""
vLLM 原子笔记生成器
专门用于vLLM双实例的原子笔记生成，支持长度分桶调度和批处理优化
"""

import asyncio
import hashlib
import os
import time
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from config import config
from tqdm import tqdm

from utils import FileUtils

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

        # 进度与缓存目录
        work_dir = config.get('storage.work_dir', '.')
        self.progress_dir = os.path.join(work_dir, 'progress', 'atomic_notes')
        FileUtils.ensure_dir(self.progress_dir)
        self.progress_index_path = os.path.join(self.progress_dir, 'processed_chunks.json')
        self.processed_chunks_map = self._load_processed_chunks()

        logger.info("VllmAtomicNoteGenerator initialized with bucket scheduler")
    
    def _init_vllm_client(self, **kwargs) -> VllmOpenAIClient:
        """初始化vLLM客户端，并进行类型与配置校验（硬失败）"""
        try:
            # 读取模型名称（如未配置则给出合理默认值）
            model_name = kwargs.get('model') or config.get('llm.note_generator.model', 'Qwen/Qwen2.5-7B-Instruct')
            endpoints = kwargs.get('endpoints') or config.get('llm.note_generator.endpoints', [])
            
            # 端点校验（必须配置至少一个端点）
            if not endpoints or not isinstance(endpoints, list):
                raise ValueError("llm.note_generator.endpoints 未配置或格式错误（需为非空列表）")
            
            # 使用工厂创建vLLM客户端（工厂内部读取配置，无需显式传参）
            client = LLMFactory.create_provider('vllm-openai')
            logger.info("vLLM client initialized via factory provider")
            
            # 类型校验：必须是 VllmOpenAIClient，否则硬失败
            if not isinstance(client, VllmOpenAIClient):
                raise TypeError(f"vLLM 客户端类型错误：期望 VllmOpenAIClient，实际为 {type(client).__name__}")
            
            # 附加日志：端点与模型一致性提示
            logger.info(f"vLLM endpoints: {endpoints}; served model alias expected: {model_name}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise
    
    def generate_atomic_notes(
        self,
        text_chunks: List[Dict[str, Any]],
        progress_tracker: Optional[Any] = None,
        on_notes_batch: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ) -> List[Dict[str, Any]]:
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
                    self._generate_atomic_notes_async(
                        text_chunks,
                        progress_tracker,
                        on_notes_batch=on_notes_batch
                    )
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
            logger.error(f"Failed to generate atomic notes: {e}")
            raise
    
    async def _generate_atomic_notes_async(
        self,
        text_chunks: List[Dict[str, Any]],
        progress_tracker: Optional[Any] = None,
        on_notes_batch: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        异步生成原子笔记，使用分桶调度批处理请求
        """
        total_chunks = len(text_chunks)
        progress_bar = tqdm(
            total=total_chunks,
            desc="Atomic notes",
            unit="chunk",
            dynamic_ncols=True,
            leave=True,
        )
        progress_lock = asyncio.Lock()

        # 统计桶分布
        results: List[Dict[str, Any]] = [None] * len(text_chunks)
        chunk_keys: Dict[int, str] = {}
        for idx, chunk in enumerate(text_chunks):
            chunk_key = self._make_chunk_key(chunk)
            chunk_keys[idx] = chunk_key
            # 如果已经存在缓存的笔记，直接加载
            if chunk_key and chunk_key in self.processed_chunks_map:
                cached_path = self.processed_chunks_map.get(chunk_key)
                if cached_path and os.path.exists(cached_path):
                    try:
                        cached_notes = FileUtils.read_json(cached_path)
                    except Exception:
                        cached_notes = []
                    results[idx] = {
                        'chunk_index': idx,
                        'notes': cached_notes if isinstance(cached_notes, list) else []
                    }
                    progress_bar.update(1)
                    continue

            prompt = self._format_atomic_note_prompt(chunk)
            # 将原始索引写入调度项，便于结果映射
            bucket_id = self.bucket_scheduler.add_request(
                prompt,
                self._get_atomic_note_system_prompt(),
                index=idx,
                chunk_key=chunk_key,
            )
            self.vllm_stats['bucket_distribution'][bucket_id] = self.vllm_stats['bucket_distribution'].get(bucket_id, 0) + 1

        # 批量并发处理
        
        async def process_bucket(bucket_id: int, items):
            # items 为 BucketRequest 列表
            prompts = [it.prompt for it in items]
            system_prompt = self._get_atomic_note_system_prompt()
            
            responses: List[str] = await self.vllm_client.chat_many(prompts, system_prompt)
            
            # 处理响应
            for i, item in enumerate(items):
                idx = item.kwargs.get('index', 0)
                chunk_key = item.kwargs.get('chunk_key')
                response = responses[i] if i < len(responses) else ""
                try:
                    notes = self._process_llm_response(response, text_chunks[idx])
                except Exception:
                    notes = []
                results[idx] = {
                    'chunk_index': idx,
                    'notes': notes
                }
                if on_notes_batch and notes:
                    on_notes_batch(notes)
                await self._store_chunk_notes(chunk_key, notes, progress_lock)
                progress_bar.update(1)

        # 为每个桶创建任务
        tasks: List[Any] = []
        for bucket_id, items in self.bucket_scheduler.buckets.items():
            if not items:
                continue
            tasks.append(process_bucket(bucket_id, items))
        
        # 并发执行
        try:
            await asyncio.gather(*tasks)
        finally:
            progress_bar.close()
        
        # 扁平化各chunk的笔记列表为统一输出
        flat_notes: List[Dict[str, Any]] = []
        for item in results:
            if not item:
                continue
            note_list = item.get('notes') if isinstance(item, dict) else None
            if isinstance(note_list, list):
                for n in note_list:
                    if isinstance(n, dict):
                        flat_notes.append(n)
        
        return flat_notes

    def _make_chunk_key(self, chunk: Dict[str, Any]) -> Optional[str]:
        source_info = chunk.get('source_info') or {}
        file_path = source_info.get('file_path')
        chunk_idx = chunk.get('chunk_index')
        if not file_path:
            return None
        return f"{file_path}#{chunk_idx if chunk_idx is not None else 0}"

    def _load_processed_chunks(self) -> Dict[str, str]:
        if os.path.exists(self.progress_index_path):
            try:
                data = FileUtils.read_json(self.progress_index_path)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                logger.warning(f"Failed to load processed chunk index: {exc}")
        return {}

    async def _store_chunk_notes(self, chunk_key: Optional[str], notes: List[Dict[str, Any]], lock: asyncio.Lock):
        if not chunk_key:
            return
        chunk_hash = hashlib.md5(chunk_key.encode('utf-8')).hexdigest()
        chunk_file = os.path.join(self.progress_dir, f"{chunk_hash}.json")
        try:
            FileUtils.write_json(notes, chunk_file)
        except Exception as exc:
            logger.warning(f"Failed to write chunk notes for {chunk_key}: {exc}")
            return

        async with lock:
            self.processed_chunks_map[chunk_key] = chunk_file
            try:
                FileUtils.write_json(self.processed_chunks_map, self.progress_index_path)
            except Exception as exc:
                logger.warning(f"Failed to update chunk progress index: {exc}")
    
    def _process_llm_response(self, response: str, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析LLM响应并转换为原子笔记格式"""
        try:
            content = extract_json_from_response(response)
        except Exception:
            content = response
        
        try:
            parsed = parse_notes_response(content)
        except Exception:
            parsed = []
        
        valid_notes = filter_valid_notes(parsed)
        
        converted: List[Dict[str, Any]] = []
        for note in valid_notes:
            converted.append(self._convert_to_atomic_note_format(note, chunk_data))
        return converted
    
    def _format_atomic_note_prompt(self, chunk_data: Dict[str, Any]) -> str:
        """根据模板格式化提示"""
        system_prompt, user_prompt_tmpl = self._get_atomic_note_prompts()
        text = chunk_data.get('text', '')
        formatted_ids = self._format_sent_ids(chunk_data.get('sentence_ids'))
        # 支持两种占位符名：{text} 和 {chunk_text}
        mapping = {
            'text': text,
            'chunk_text': text,
            'sent_ids': formatted_ids
        }
        prompt = user_prompt_tmpl.format(**mapping)
        return prompt
    
    async def health_check(self) -> Dict[str, Any]:
        """返回vLLM客户端和分桶调度器的健康状态"""
        try:
            vllm_health = await self.vllm_client.health_check()
            status = 'healthy'
            if not vllm_health:
                status = 'degraded'
            return {
                'status': status,
                'vllm_endpoints': vllm_health,
                'bucket_scheduler': {
                    'bucket_stats': self.bucket_scheduler.get_bucket_stats()
                }
            }
        except Exception:
            return {
                'status': 'error',
                'vllm_endpoints': {},
                'bucket_scheduler': {
                    'bucket_stats': self.bucket_scheduler.get_bucket_stats()
                }
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """返回性能统计（请求分布、吞吐等）"""
        total = sum(self.vllm_stats['bucket_distribution'].values()) or 1
        return {
            'total_requests': self.vllm_stats['total_requests'],
            'bucket_distribution': self.vllm_stats['bucket_distribution'],
            'processing_time': self.vllm_stats['processing_time'],
            'throughput_qps': self.vllm_stats['throughput_qps'],
            'avg_requests_per_bucket': total / max(len(self.vllm_stats['bucket_distribution']) or 1, 1)
        }
    
    async def close(self):
        try:
            await self.vllm_client.close()
        except Exception:
            pass
    
    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 避免未等待协程的警告：不在析构中调度关闭
                pass
            else:
                # 避免在析构中运行事件循环
                pass
        except Exception:
            pass


def create_vllm_note_generator(**kwargs) -> VllmAtomicNoteGenerator:
    """工厂方法：创建vLLM原子笔记生成器"""
    try:
        gen = VllmAtomicNoteGenerator(**kwargs)
        return gen
    except Exception as e:
        logger.error(f"Failed to create VllmAtomicNoteGenerator: {e}")
        raise
