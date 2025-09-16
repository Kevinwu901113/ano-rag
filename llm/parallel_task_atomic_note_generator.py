from typing import List, Dict, Any, Union, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json
from .atomic_note_generator import AtomicNoteGenerator
from .local_llm import LocalLLM
from .ollama_client import OllamaClient
from .lmstudio_client import LMStudioClient
from utils.json_utils import extract_json_from_response
from config import config
from .prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)

class ParallelTaskAtomicNoteGenerator(AtomicNoteGenerator):
    """并行任务分配原子笔记生成器，支持Ollama和LM Studio的任务分配处理"""
    
    def __init__(self, llm: LocalLLM = None):
        super().__init__(llm)
        
        # 加载并行处理配置
        self.parallel_config = config.get('atomic_note_generation', {})
        self.parallel_enabled = self.parallel_config.get('parallel_enabled', False)
        self.parallel_strategy = self.parallel_config.get('parallel_strategy', 'task_division')
        
        # 任务分配配置
        self.task_division_config = self.parallel_config.get('task_division', {})
        self.allocation_method = self.task_division_config.get('allocation_method', 'round_robin')
        self.enable_fallback = self.task_division_config.get('enable_fallback', True)
        self.fallback_timeout = self.task_division_config.get('fallback_timeout', 10)
        
        # 初始化客户端
        self.ollama_client = None
        self.lmstudio_client = None
        self._init_parallel_clients()
        
        # 性能监控
        self.monitoring_config = self.parallel_config.get('monitoring', {})
        self.monitoring_enabled = self.monitoring_config.get('enabled', True)
        self.stats = {
            'ollama_tasks': 0,
            'lmstudio_tasks': 0,
            'ollama_success': 0,
            'lmstudio_success': 0,
            'ollama_errors': 0,
            'lmstudio_errors': 0,
            'ollama_total_time': 0.0,
            'lmstudio_total_time': 0.0,
            'fallback_count': 0
        }
        self._stats_lock = threading.Lock()
        
    def _init_parallel_clients(self):
        """初始化并行处理客户端"""
        if not self.parallel_enabled or self.parallel_strategy != 'task_division':
            return
            
        try:
            # 初始化Ollama客户端
            ollama_config = self.parallel_config.get('ollama', {})
            if ollama_config:
                self.ollama_client = OllamaClient(
                    base_url=ollama_config.get('base_url', 'http://localhost:11434'),
                    model=ollama_config.get('model', 'qwen2.5:latest')
                )
                logger.info(f"Initialized Ollama client: {ollama_config.get('model')}")
            
            # 初始化LM Studio客户端
            lmstudio_config = self.parallel_config.get('lmstudio', {})
            if lmstudio_config:
                self.lmstudio_client = LMStudioClient(
                    base_url=lmstudio_config.get('base_url', 'http://localhost:1234/v1'),
                    model=lmstudio_config.get('model', 'qwen2.5:latest'),
                    port=lmstudio_config.get('port', 1234)
                )
                logger.info(f"Initialized LM Studio client: {lmstudio_config.get('model')}")
                
        except Exception as e:
            logger.error(f"Failed to initialize parallel clients: {e}")
            self.parallel_enabled = False
    
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """生成原子笔记的主入口，支持并行任务分配"""
        if not self.parallel_enabled or not (self.ollama_client and self.lmstudio_client):
            logger.info("Parallel processing disabled, falling back to original implementation")
            return super().generate_atomic_notes(text_chunks, progress_tracker)
        
        logger.info(f"Starting parallel task division processing for {len(text_chunks)} chunks")
        return self._generate_atomic_notes_parallel_task_division(text_chunks, progress_tracker)
    
    def _generate_atomic_notes_parallel_task_division(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """使用任务分配策略并行生成原子笔记"""
        system_prompt = self._get_atomic_note_system_prompt()
        atomic_notes = [None] * len(text_chunks)
        
        # 任务分配
        ollama_tasks, lmstudio_tasks = self._allocate_tasks(text_chunks)
        
        logger.info(f"Task allocation: Ollama={len(ollama_tasks)}, LM Studio={len(lmstudio_tasks)}")
        
        # 并行处理 - 动态调整并发数量以确保真正并行
        # 计算合适的并发数：确保ollama和lmstudio的任务都能同时开始
        ollama_count = len(ollama_tasks)
        lmstudio_count = len(lmstudio_tasks)
        
        # 至少需要能同时运行两种客户端的任务
        min_workers_needed = min(ollama_count, 1) + min(lmstudio_count, 1)
        # 但也不要超过总任务数或系统限制
        max_workers = min(len(text_chunks), max(min_workers_needed, 4))
        
        logger.info(f"Using {max_workers} workers for {len(text_chunks)} tasks (Ollama: {ollama_count}, LM Studio: {lmstudio_count})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # 同时提交所有任务，确保真正并行执行
            # 交替提交Ollama和LM Studio任务，避免顺序执行
            all_tasks = []
            
            # 交替添加任务以确保两种客户端的任务都能尽早开始
            max_tasks = max(len(ollama_tasks), len(lmstudio_tasks))
            for i in range(max_tasks):
                # 优先添加LM Studio任务（通常较快）
                if i < len(lmstudio_tasks):
                    chunk_data, index = lmstudio_tasks[i]
                    all_tasks.append((chunk_data, index, 'lmstudio'))
                # 然后添加Ollama任务
                if i < len(ollama_tasks):
                    chunk_data, index = ollama_tasks[i]
                    all_tasks.append((chunk_data, index, 'ollama'))
            
            # 同时提交所有任务
            for chunk_data, index, client_type in all_tasks:
                if client_type == 'ollama':
                    future = executor.submit(self._process_with_ollama, chunk_data, index, system_prompt)
                else:
                    future = executor.submit(self._process_with_lmstudio, chunk_data, index, system_prompt)
                futures.append((future, index, client_type))
            
            # 使用as_completed来真正并行等待结果，而不是按顺序等待
            completed_count = 0
            # 创建future到元数据的映射
            future_to_meta = {f[0]: (f[1], f[2]) for f in futures}
            
            for future in as_completed([f[0] for f in futures]):
                try:
                    # 从映射中获取index和client_type
                    index, client_type = future_to_meta[future]
                    
                    # 设置超时时间 - 从配置获取，默认120秒
                    timeout_value = self.parallel_config.get('fallback_timeout', 120)
                    logger.debug(f"Waiting for {client_type} task {index} with {timeout_value}s timeout")
                    
                    note = future.result(timeout=timeout_value)
                    atomic_notes[index] = note
                    
                    with self._stats_lock:
                        if client_type == 'ollama':
                            self.stats['ollama_success'] += 1
                        else:
                            self.stats['lmstudio_success'] += 1
                    
                    completed_count += 1
                    if progress_tracker:
                        progress_tracker.update(1)
                        
                except Exception as e:
                    # 从映射中获取index和client_type（如果future已经完成但有异常）
                    index, client_type = future_to_meta.get(future, (None, 'unknown'))
                    
                    # 获取详细的错误信息
                    error_type = type(e).__name__
                    error_msg = str(e) if str(e) else f"Unknown error in {client_type} processing"
                    
                    # 特殊处理TimeoutError
                    if isinstance(e, TimeoutError):
                        error_msg = f"Task timeout after {self.fallback_timeout}s - consider increasing fallback_timeout in config"
                        logger.error(f"Task failed for index {index} using {client_type}: [{error_type}] {error_msg}")
                    else:
                        logger.error(f"Task failed for index {index} using {client_type}: [{error_type}] {error_msg}")
                    
                    # 获取完整的traceback信息
                    import traceback
                    tb_str = traceback.format_exc()
                    logger.error(f"Full traceback for {client_type} task {index}:\n{tb_str}")
                    
                    with self._stats_lock:
                        if client_type == 'ollama':
                            self.stats['ollama_errors'] += 1
                        else:
                            self.stats['lmstudio_errors'] += 1
                    
                    # 失败回退处理
                    if self.enable_fallback:
                        try:
                            fallback_note = self._fallback_process(text_chunks[index], system_prompt, client_type)
                            atomic_notes[index] = fallback_note
                            with self._stats_lock:
                                self.stats['fallback_count'] += 1
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed for index {index}: {fallback_error}")
                            atomic_notes[index] = self._create_fallback_note(text_chunks[index])
                    else:
                        atomic_notes[index] = self._create_fallback_note(text_chunks[index])
                    
                    completed_count += 1
                    if progress_tracker:
                        progress_tracker.update(1)
        
        # 后处理
        for i, note in enumerate(atomic_notes):
            if note is None:
                atomic_notes[i] = self._create_fallback_note(text_chunks[i] if i < len(text_chunks) else {'text': ''})
            
            if not isinstance(note, dict):
                atomic_notes[i] = {'content': str(note), 'error': True}
            
            atomic_notes[i]['note_id'] = f"note_{i:06d}"
            atomic_notes[i]['created_at'] = self._get_timestamp()
        
        # 记录统计信息
        if self.monitoring_enabled:
            self._log_performance_stats()
        
        logger.info(f"Parallel task division completed: {len(atomic_notes)} notes generated")
        return atomic_notes
    
    def _allocate_tasks(self, text_chunks: List[Dict[str, Any]]) -> tuple:
        """分配任务给Ollama和LM Studio"""
        ollama_tasks = []
        lmstudio_tasks = []
        
        if self.allocation_method == 'round_robin':
            # 轮询分配：偶数索引给Ollama，奇数索引给LM Studio
            for i, chunk in enumerate(text_chunks):
                if i % 2 == 0:
                    ollama_tasks.append((chunk, i))
                else:
                    lmstudio_tasks.append((chunk, i))
        
        elif self.allocation_method == 'batch_split':
            # 批次分割：前一半给Ollama，后一半给LM Studio
            mid_point = len(text_chunks) // 2
            for i, chunk in enumerate(text_chunks):
                if i < mid_point:
                    ollama_tasks.append((chunk, i))
                else:
                    lmstudio_tasks.append((chunk, i))
        
        # 更新统计信息
        with self._stats_lock:
            self.stats['ollama_tasks'] += len(ollama_tasks)
            self.stats['lmstudio_tasks'] += len(lmstudio_tasks)
        
        return ollama_tasks, lmstudio_tasks
    
    def _process_with_ollama(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Dict[str, Any]:
        """使用Ollama处理单个任务"""
        start_time = time.time()
        try:
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            logger.debug(f"Ollama processing task {index}, text length: {len(text)}")
            
            # 添加超时控制
            # 从配置获取超时时间，默认90秒
            timeout = self.parallel_config.get('ollama', {}).get('timeout', 90)
            response = self.ollama_client.generate(prompt, system_prompt, timeout=timeout)
            
            # 检查响应是否为空
            if not response or response.strip() == "":
                logger.error(f"Ollama task {index}: Empty response received")
                raise ValueError("Ollama returned empty response - possible connection or model issue")
            
            logger.debug(f"Ollama task {index}: Response length: {len(response)}")
            
            # 解析响应
            cleaned_response = extract_json_from_response(response)
            if not cleaned_response:
                logger.error(f"Ollama task {index}: No valid JSON in response: {response[:200]}...")
                raise ValueError(f"No valid JSON found in Ollama response: {response[:200]}...")
            
            note_data = json.loads(cleaned_response)
            if isinstance(note_data, list) and note_data:
                note_data = note_data[0]
            
            result = self._create_atomic_note_from_data(note_data, chunk_data)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['ollama_total_time'] += processing_time
            
            logger.debug(f"Ollama task {index}: Completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['ollama_total_time'] += processing_time
            
            # 详细的错误日志
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                logger.error(f"Ollama task {index}: Timeout after {processing_time:.2f}s - {e}")
            elif "connection" in error_msg.lower():
                logger.error(f"Ollama task {index}: Connection error - {e}")
            elif "json" in error_msg.lower():
                logger.error(f"Ollama task {index}: JSON parsing error - {e}")
            else:
                logger.error(f"Ollama task {index}: Unknown error after {processing_time:.2f}s - {e}")
            
            raise e
        except InterruptedError as e:
            # Handle graceful shutdown
            processing_time = time.time() - start_time
            logger.warning(f"Ollama task {index}: Interrupted due to shutdown after {processing_time:.2f}s - {e}")
            # Return a default result instead of raising to allow graceful shutdown
            return {
                'chunk_id': chunk_data.get('chunk_id', f'chunk_{index}'),
                'atomic_notes': [],
                'error': 'interrupted_by_shutdown',
                'processing_time': processing_time
            }
    
    def _process_with_lmstudio(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Dict[str, Any]:
        """使用LM Studio处理单个任务"""
        start_time = time.time()
        try:
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            response = self.lmstudio_client.generate(prompt, system_prompt)
            
            # 解析响应
            cleaned_response = extract_json_from_response(response)
            if not cleaned_response:
                raise ValueError(f"No valid JSON found in LM Studio response")
            
            note_data = json.loads(cleaned_response)
            if isinstance(note_data, list) and note_data:
                note_data = note_data[0]
            
            result = self._create_atomic_note_from_data(note_data, chunk_data)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
            raise e
    
    def _fallback_process(self, chunk_data: Dict[str, Any], system_prompt: str, failed_client: str) -> Dict[str, Any]:
        """回退处理机制，包含重试逻辑"""
        logger.info(f"Attempting fallback for failed {failed_client} task")
        
        # 首先尝试重试原客户端（可能是临时网络问题）
        max_retries = 2
        for retry in range(max_retries):
            try:
                logger.info(f"Retry {retry + 1}/{max_retries} with {failed_client}")
                if failed_client == 'ollama' and self.ollama_client:
                    return self._process_with_ollama(chunk_data, -1, system_prompt)
                elif failed_client == 'lmstudio' and self.lmstudio_client:
                    return self._process_with_lmstudio(chunk_data, -1, system_prompt)
            except Exception as e:
                logger.warning(f"Retry {retry + 1} failed: {e}")
                if retry == max_retries - 1:
                    break
        
        # 选择备用客户端
        try:
            if failed_client == 'ollama' and self.lmstudio_client:
                logger.info("Switching to LM Studio as fallback")
                return self._process_with_lmstudio(chunk_data, -1, system_prompt)
            elif failed_client == 'lmstudio' and self.ollama_client:
                logger.info("Switching to Ollama as fallback")
                return self._process_with_ollama(chunk_data, -1, system_prompt)
        except Exception as e:
            logger.error(f"Fallback client also failed: {e}")
        
        # 最后使用原始LLM
        logger.warning("All parallel clients failed, using original LLM")
        return self._fallback_to_original_llm(chunk_data, system_prompt)
    
    def _fallback_to_original_llm(self, chunk_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """使用原始LLM作为最终回退"""
        try:
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            response = self.llm.generate(prompt, system_prompt)
            
            # 解析响应
            cleaned_response = extract_json_from_response(response)
            if not cleaned_response:
                raise ValueError(f"No valid JSON found in original LLM response")
            
            note_data = json.loads(cleaned_response)
            if isinstance(note_data, list) and note_data:
                note_data = note_data[0]
            
            return self._create_atomic_note_from_data(note_data, chunk_data)
            
        except Exception as e:
            logger.error(f"Original LLM fallback also failed: {e}")
            # 返回一个基本的原子笔记结构
            return {
                'title': f"Failed to process chunk {chunk_data.get('chunk_id', 'unknown')}",
                'content': chunk_data.get('text', '')[:200] + '...',
                'note_id': f"fallback_{chunk_data.get('chunk_id', 'unknown')}",
                'source': chunk_data.get('source', 'unknown'),
                'error': str(e)
            }
    
    def _log_performance_stats(self):
        """记录性能统计信息"""
        with self._stats_lock:
            stats = self.stats.copy()
        
        logger.info("=== Parallel Processing Performance Stats ===")
        logger.info(f"Ollama: {stats['ollama_tasks']} tasks, {stats['ollama_success']} success, {stats['ollama_errors']} errors")
        logger.info(f"LM Studio: {stats['lmstudio_tasks']} tasks, {stats['lmstudio_success']} success, {stats['lmstudio_errors']} errors")
        logger.info(f"Fallback count: {stats['fallback_count']}")
        
        if stats['ollama_tasks'] > 0:
            avg_ollama_time = stats['ollama_total_time'] / stats['ollama_tasks']
            logger.info(f"Ollama average time: {avg_ollama_time:.2f}s")
        
        if stats['lmstudio_tasks'] > 0:
            avg_lmstudio_time = stats['lmstudio_total_time'] / stats['lmstudio_tasks']
            logger.info(f"LM Studio average time: {avg_lmstudio_time:.2f}s")
        
        logger.info("=============================================")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self._stats_lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        with self._stats_lock:
            self.stats = {
                'ollama_tasks': 0,
                'lmstudio_tasks': 0,
                'ollama_success': 0,
                'lmstudio_success': 0,
                'ollama_errors': 0,
                'lmstudio_errors': 0,
                'ollama_total_time': 0.0,
                'lmstudio_total_time': 0.0,
                'fallback_count': 0
            }