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
from utils.robust_json_parser import extract_json_array
from utils.performance_monitor import (
    get_monitor, record_task_performance, record_fallback_operation,
    AlertThresholds, setup_monitor
)
from config import config
from .prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)

class ParallelTaskAtomicNoteGenerator(AtomicNoteGenerator):
    """并行任务原子笔记生成器，支持Ollama和LM Studio双客户端"""
    
    def __init__(self, llm: LocalLLM = None):
        super().__init__(llm)
        self.parallel_config = config.get('atomic_note_generation', {})
        self.ollama_client = None
        self.lmstudio_client = None
        self._stats_lock = threading.Lock()
        
        # 初始化并行客户端
        self._init_parallel_clients()
        
        # 性能统计
        self.stats = {
            'ollama_tasks': 0,
            'lmstudio_tasks': 0,
            'ollama_total_time': 0.0,
            'lmstudio_total_time': 0.0,
            'ollama_errors': 0,
            'lmstudio_errors': 0,
            'fallback_count': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'parse_failures': 0,
            'retry_attempts': 0,
            'empty_results': 0
        }
        
        # 设置性能监控器
        monitoring_config = self.parallel_config.get('monitoring', {})
        thresholds = AlertThresholds(
            error_rate_threshold=monitoring_config.get('error_rate_threshold', 0.1),
            avg_duration_threshold=monitoring_config.get('avg_duration_threshold', 30.0),
            fallback_rate_threshold=monitoring_config.get('fallback_rate_threshold', 0.2),
            consecutive_failures_threshold=monitoring_config.get('consecutive_failures_threshold', 5),
            monitoring_window_minutes=monitoring_config.get('monitoring_window_minutes', 10)
        )
        self.monitor = setup_monitor(thresholds)
        
        # 添加告警回调
        self.monitor.add_alert_callback(self._handle_performance_alert)
        
        logger.info("ParallelTaskAtomicNoteGenerator initialized with performance monitoring")
    
    def _init_parallel_clients(self):
        """初始化Ollama和LM Studio客户端"""
        try:
            # 初始化Ollama客户端
            ollama_config = self.parallel_config.get('ollama', {})
            if ollama_config.get('enabled', True):
                # 使用配置文件中的设置，而不是硬编码的默认值
                self.ollama_client = OllamaClient(
                    base_url=ollama_config.get('base_url'),
                    model=ollama_config.get('model')
                )
                # timeout 参数由 OllamaClient 内部从配置文件读取，不需要在构造函数中传递
                logger.info(f"Ollama client initialized: {ollama_config.get('model', 'default')}")
                
                # 测试Ollama连接
                if not self.ollama_client.is_available():
                    logger.error("Ollama client is not available")
                    self.ollama_client = None
                else:
                    logger.info("Ollama client connection verified")
            else:
                logger.info("Ollama client disabled in configuration")
            
            # 初始化LM Studio客户端
            lmstudio_config = self.parallel_config.get('lmstudio', {})
            if lmstudio_config.get('enabled', False):
                # 使用配置文件中的设置，而不是硬编码的默认值
                self.lmstudio_client = LMStudioClient(
                    base_url=lmstudio_config.get('base_url'),
                    model=lmstudio_config.get('model')
                )
                logger.info(f"LM Studio client initialized: {lmstudio_config.get('model', 'default')}")
                
                # 测试LM Studio连接
                if not self.lmstudio_client.is_available():
                    logger.error("LM Studio client is not available")
                    self.lmstudio_client = None
                else:
                    logger.info("LM Studio client connection verified")
            else:
                logger.info("LM Studio client disabled in configuration")
                
        except Exception as e:
            logger.error(f"Failed to initialize parallel clients: {e}")
            # 如果初始化失败，回退到原始LLM
    
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """生成原子笔记的主入口方法"""
        if not self.ollama_client and not self.lmstudio_client:
            logger.warning("No parallel clients available, falling back to original LLM")
            return super().generate_atomic_notes(text_chunks, progress_tracker)
        
        return self._generate_atomic_notes_parallel_task_division(text_chunks, progress_tracker)
    
    def _generate_atomic_notes_parallel_task_division(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """使用任务分配的并行处理方式"""
        if not text_chunks:
            return []
        
        with self._stats_lock:
            self.stats['total_chunks'] = len(text_chunks)
        
        logger.info(f"Starting parallel processing of {len(text_chunks)} chunks")
        
        # 分配任务到不同的客户端
        ollama_tasks, lmstudio_tasks = self._allocate_tasks(text_chunks)
        
        results = []
        system_prompt = ATOMIC_NOTEGEN_SYSTEM_PROMPT
        
        # 使用ThreadPoolExecutor进行并行处理
        max_workers = self.parallel_config.get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_info = {}
            
            # 提交Ollama任务
            for chunk_data, index in ollama_tasks:
                future = executor.submit(self._process_with_ollama, chunk_data, index, system_prompt)
                future_to_info[future] = ('ollama', index, chunk_data)
            
            # 提交LM Studio任务
            for chunk_data, index in lmstudio_tasks:
                future = executor.submit(self._process_with_lmstudio, chunk_data, index, system_prompt)
                future_to_info[future] = ('lmstudio', index, chunk_data)
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_info):
                client_type, index, chunk_data = future_to_info[future]
                
                try:
                    result = future.result()
                    if result and result.get('atomic_notes'):
                        with self._stats_lock:
                            self.stats['successful_chunks'] += 1
                    results.append((index, result))
                    
                except Exception as e:
                    logger.error(f"Task {index} ({client_type}) failed: {e}")
                    # 创建空结果
                    empty_result = self._create_empty_atomic_note(chunk_data)
                    results.append((index, empty_result))
                
                completed_count += 1
                if progress_tracker:
                    progress_tracker.update(completed_count / len(text_chunks))
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        # 记录性能统计
        self._log_performance_stats()
        
        return final_results
    
    def _allocate_tasks(self, text_chunks: List[Dict[str, Any]]) -> tuple:
        """将任务分配给不同的客户端"""
        ollama_tasks = []
        lmstudio_tasks = []
        
        # 获取分配策略
        allocation_strategy = self.parallel_config.get('allocation_strategy', 'round_robin')
        
        # 调试日志：检查客户端状态
        logger.info(f"Client status - Ollama: {'Available' if self.ollama_client else 'None'}, "
                   f"LMStudio: {'Available' if self.lmstudio_client else 'None'}")
        
        if allocation_strategy == 'round_robin':
            # 轮询分配
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if i % 2 == 0:
                    # 偶数索引优先分配给Ollama
                    if self.ollama_client:
                        ollama_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} assigned to Ollama")
                    elif self.lmstudio_client:
                        lmstudio_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} fallback to LMStudio (Ollama unavailable)")
                    else:
                        logger.warning(f"Task {i} cannot be assigned - no clients available")
                else:
                    # 奇数索引优先分配给LMStudio
                    if self.lmstudio_client:
                        lmstudio_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} assigned to LMStudio")
                    elif self.ollama_client:
                        ollama_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} fallback to Ollama (LMStudio unavailable)")
                    else:
                        logger.warning(f"Task {i} cannot be assigned - no clients available")
        
        elif allocation_strategy == 'ollama_first':
            # 优先使用Ollama
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if self.ollama_client:
                    ollama_tasks.append((chunk_with_id, i))
                elif self.lmstudio_client:
                    lmstudio_tasks.append((chunk_with_id, i))
        
        elif allocation_strategy == 'lmstudio_first':
            # 优先使用LM Studio
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if self.lmstudio_client:
                    lmstudio_tasks.append((chunk_with_id, i))
                elif self.ollama_client:
                    ollama_tasks.append((chunk_with_id, i))
        
        logger.info(f"Task allocation completed - Ollama: {len(ollama_tasks)} tasks, LMStudio: {len(lmstudio_tasks)} tasks")
        
        # 更新统计
        with self._stats_lock:
            self.stats['ollama_tasks'] += len(ollama_tasks)
            self.stats['lmstudio_tasks'] += len(lmstudio_tasks)
        
        return ollama_tasks, lmstudio_tasks
    
    def _process_with_ollama(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Dict[str, Any]:
        """使用Ollama处理单个任务，包含鲁棒解析和重试机制"""
        start_time = time.time()
        task_id = f"ollama_task_{index}"
        
        try:
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            logger.debug(f"Ollama processing task {index}, text length: {len(text)}")
            
            # 第一次尝试：正常处理
            timeout = self.parallel_config.get('ollama', {}).get('timeout', 90)
            response = self.ollama_client.generate(prompt, system_prompt, timeout=timeout)
            
            # 检查响应是否为空
            if not response or response.strip() == "":
                logger.error(f"Ollama task {index}: Empty response received")
                raise ValueError("Ollama returned empty response - possible connection or model issue")
            
            logger.debug(f"Ollama task {index}: Response length: {len(response)}")
            
            # 第一次解析尝试
            parsed_data, error_msg = self._robust_parse_response(response)
            
            if parsed_data is not None:
                result = self._create_atomic_note_from_parsed_data(parsed_data, chunk_data)
                processing_time = time.time() - start_time
                with self._stats_lock:
                    self.stats['ollama_total_time'] += processing_time
                
                # 记录成功的任务性能
                record_task_performance(
                    task_type="ollama",
                    duration=processing_time,
                    success=True,
                    chunk_size=len(text)
                )
                
                logger.debug(f"Ollama task {index}: Completed in {processing_time:.2f}s")
                return result
            
            # 第一次解析失败，尝试重试
            logger.warning(f"Ollama task {index} first parse failed: {error_msg}")
            with self._stats_lock:
                self.stats['parse_failures'] += 1
                self.stats['retry_attempts'] += 1
            
            # 第二次尝试：使用更严格的提示
            strict_system = system_prompt + "\n\n重要：你必须仅输出JSON数组，不得有任何其他文字！"
            strict_prompt = f"请严格按照JSON数组格式输出结果：\n{prompt}"
            
            response = self.ollama_client.generate(strict_prompt, strict_system, timeout=timeout)
            parsed_data, error_msg = self._robust_parse_response(response)
            
            if parsed_data is not None:
                result = self._create_atomic_note_from_parsed_data(parsed_data, chunk_data)
                processing_time = time.time() - start_time
                with self._stats_lock:
                    self.stats['ollama_total_time'] += processing_time
                
                # 记录重试成功的任务性能
                record_task_performance(
                    task_type="ollama_retry",
                    duration=processing_time,
                    success=True,
                    chunk_size=len(text)
                )
                
                logger.debug(f"Ollama task {index}: Retry completed in {processing_time:.2f}s")
                return result
            
            # 两次尝试都失败，回退到LM Studio
            logger.warning(f"Ollama task {index} retry also failed: {error_msg}, falling back to LM Studio")
            with self._stats_lock:
                self.stats['fallback_count'] += 1
            
            # 记录回退操作
            record_fallback_operation(
                from_client="ollama",
                to_client="lmstudio",
                success=False
            )
            
            if self.lmstudio_client:
                return self._process_with_lmstudio(chunk_data, index, system_prompt)
            
            # 没有LM Studio可用，返回空结果
            logger.error(f"Both Ollama and LM Studio failed for task {index}, returning empty result")
            with self._stats_lock:
                self.stats['empty_results'] += 1
            
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['ollama_total_time'] += processing_time
            
            # 记录失败的任务性能
            record_task_performance(
                task_type="ollama",
                duration=processing_time,
                success=False,
                chunk_size=len(text)
            )
            
            return self._create_empty_atomic_note(chunk_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['ollama_total_time'] += processing_time
                self.stats['ollama_errors'] += 1
            
            # 详细的错误日志
            error_msg = str(e)
            error_type = "unknown_error"
            if "timeout" in error_msg.lower():
                error_type = "timeout"
                logger.error(f"Ollama task {index}: Timeout after {processing_time:.2f}s - {e}")
            elif "connection" in error_msg.lower():
                error_type = "connection_error"
                logger.error(f"Ollama task {index}: Connection error - {e}")
            elif "json" in error_msg.lower():
                error_type = "json_parsing_error"
                logger.error(f"Ollama task {index}: JSON parsing error - {e}")
            else:
                logger.error(f"Ollama task {index}: Unknown error after {processing_time:.2f}s - {e}")
            
            # 记录失败的任务性能
            record_task_performance(
                task_type="ollama",
                duration=processing_time,
                success=False,
                chunk_size=len(chunk_data.get('text', ''))
            )
            
            # 不抛出异常，返回空结果让流水线继续
            return self._create_empty_atomic_note(chunk_data)
            
        except InterruptedError as e:
            # Handle graceful shutdown
            processing_time = time.time() - start_time
            logger.warning(f"Ollama task {index}: Interrupted due to shutdown after {processing_time:.2f}s - {e}")
            
            # 记录中断的任务
            record_task_performance(
                task_type="ollama",
                duration=processing_time,
                success=False,
                chunk_size=len(chunk_data.get('text', ''))
            )
            
            # Return a default result instead of raising to allow graceful shutdown
            return self._create_empty_atomic_note(chunk_data)
    
    def _robust_parse_response(self, response: str) -> tuple:
        """鲁棒的响应解析方法"""
        if not response or not response.strip():
            return None, "Empty response"
        
        # 使用增强的JSON解析器
        parsed_data, error_msg = extract_json_array(response, prefer="array")
        
        if parsed_data is not None:
            # 处理不同的数据格式
            if isinstance(parsed_data, list):
                return parsed_data, None
            elif isinstance(parsed_data, dict):
                # 尝试从字典中提取数组
                for key in ['facts', 'atomic_notes', 'notes', 'data']:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        return parsed_data[key], None
                # 如果没找到数组，将字典包装成数组
                return [parsed_data], None
            else:
                return None, f"Unexpected data type: {type(parsed_data)}"
        
        return None, error_msg or "Failed to parse JSON"
    
    def _create_atomic_note_from_parsed_data(self, parsed_data: list, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """从解析的数据创建原子笔记结果"""
        if not parsed_data:
            return self._create_empty_atomic_note(chunk_data)
        
        # 处理解析后的数据，确保格式正确
        atomic_notes = []
        for item in parsed_data:
            if isinstance(item, dict):
                atomic_notes.append(item)
            else:
                # 如果不是字典，尝试转换
                atomic_notes.append({'content': str(item)})
        
        return {
            'chunk_id': chunk_data.get('chunk_id', 'unknown'),
            'atomic_notes': atomic_notes,
            'source_text': chunk_data.get('text', ''),
            'metadata': chunk_data.get('metadata', {})
        }
    
    def _create_empty_atomic_note(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建空的原子笔记结果"""
        return {
            'chunk_id': chunk_data.get('chunk_id', 'unknown'),
            'atomic_notes': [],
            'source_text': chunk_data.get('text', ''),
            'metadata': chunk_data.get('metadata', {}),
            'error': 'parsing_failed'
        }
    
    def _process_with_lmstudio(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Dict[str, Any]:
        """使用LM Studio处理单个任务，包含鲁棒解析和重试机制"""
        start_time = time.time()
        task_id = f"lmstudio_task_{index}"
        
        try:
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            logger.debug(f"LM Studio processing task {index}, text length: {len(text)}")
            
            # 第一次尝试：启用JSON模式
            response = self.lmstudio_client.generate(prompt, system_prompt, try_json_mode=True)
            
            # 第一次解析尝试
            parsed_data, error_msg = self._robust_parse_response(response)
            
            if parsed_data is not None:
                result = self._create_atomic_note_from_parsed_data(parsed_data, chunk_data)
                processing_time = time.time() - start_time
                with self._stats_lock:
                    self.stats['lmstudio_total_time'] += processing_time
                
                # 记录成功的任务性能
                record_task_performance(
                    task_type="lmstudio",
                    duration=processing_time,
                    success=True,
                    chunk_size=len(text)
                )
                
                return result
            
            # 第一次解析失败，尝试重试
            logger.warning(f"LM Studio first parse failed: {error_msg}")
            with self._stats_lock:
                self.stats['parse_failures'] += 1
                self.stats['retry_attempts'] += 1
            
            # 第二次尝试：使用更严格的提示
            strict_system = system_prompt + "\n\n重要：你必须仅输出JSON数组，不得有任何其他文字！"
            strict_prompt = f"请严格按照JSON数组格式输出结果：\n{prompt}"
            
            response = self.lmstudio_client.generate(strict_prompt, strict_system, try_json_mode=True)
            parsed_data, error_msg = self._robust_parse_response(response)
            
            if parsed_data is not None:
                result = self._create_atomic_note_from_parsed_data(parsed_data, chunk_data)
                processing_time = time.time() - start_time
                with self._stats_lock:
                    self.stats['lmstudio_total_time'] += processing_time
                
                # 记录重试成功的任务性能
                record_task_performance(
                    task_type="lmstudio",
                    duration=processing_time,
                    success=True,
                    retry_count=1,
                    chunk_size=len(text)
                )
                
                return result
            
            # 两次尝试都失败，回退到Ollama
            logger.warning(f"LM Studio retry also failed: {error_msg}, falling back to Ollama")
            with self._stats_lock:
                self.stats['fallback_count'] += 1
            
            # 记录回退操作
            record_fallback_operation(
                from_client="lmstudio",
                to_client="ollama",
                success=False
            )
            
            if self.ollama_client:
                return self._process_with_ollama(chunk_data, index, system_prompt)
            
            # 没有Ollama可用，返回空结果
            logger.error("Both LM Studio and Ollama failed, returning empty result")
            with self._stats_lock:
                self.stats['empty_results'] += 1
            
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
            
            # 记录失败的任务性能
            record_task_performance(
                task_type="lmstudio",
                duration=processing_time,
                success=False,
                chunk_size=len(text)
            )
            
            return self._create_empty_atomic_note(chunk_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
                self.stats['lmstudio_errors'] += 1
            
            # 确定错误类型
            error_msg = str(e)
            error_type = "unknown_error"
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "connection" in error_msg.lower():
                error_type = "connection_error"
            elif "json" in error_msg.lower():
                error_type = "json_parsing_error"
            
            logger.error(f"LM Studio processing error: {e}")
            
            # 记录失败的任务性能
            record_task_performance(
                task_type="lmstudio",
                duration=processing_time,
                success=False,
                chunk_size=len(chunk_data.get('text', ''))
            )
            
            # 不抛出异常，返回空结果让流水线继续
            return self._create_empty_atomic_note(chunk_data)
    
    def _fallback_process(self, chunk_data: Dict[str, Any], system_prompt: str, failed_client: str) -> Dict[str, Any]:
        """回退处理机制，包含重试逻辑"""
        start_time = time.time()
        logger.info(f"Attempting fallback for failed {failed_client} task")
        
        try:
            # 首先尝试重试原客户端（可能是临时网络问题）
            if failed_client == 'ollama' and self.lmstudio_client:
                result = self._process_with_lmstudio(chunk_data, -1, system_prompt)
                
                # 记录成功的回退操作
                record_fallback_operation(
                    from_client="ollama",
                    to_client="lmstudio",
                    success=True
                )
                
                return result
                
            elif failed_client == 'lmstudio' and self.ollama_client:
                result = self._process_with_ollama(chunk_data, -1, system_prompt)
                
                # 记录成功的回退操作
                record_fallback_operation(
                    from_client="lmstudio",
                    to_client="ollama",
                    success=True
                )
                
                return result
            else:
                # 如果没有其他客户端可用，回退到原始LLM
                result = self._fallback_to_original_llm(chunk_data, system_prompt)
                
                # 记录回退到原始LLM的操作
                record_fallback_operation(
                    from_client=failed_client,
                    to_client="original_llm",
                    success=True
                )
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Fallback process failed: {e}")
            
            # 记录失败的回退操作
            record_fallback_operation(
                from_client=failed_client,
                to_client="fallback",
                success=False
            )
            
            # 记录失败的任务性能
            record_task_performance(
                task_type=f"{failed_client}_fallback",
                duration=processing_time,
                success=False,
                chunk_size=len(chunk_data.get('text', ''))
            )
            
            return self._create_empty_atomic_note(chunk_data)
    
    def _fallback_to_original_llm(self, chunk_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """回退到原始LLM处理"""
        start_time = time.time()
        
        try:
            logger.info("Falling back to original LLM")
            text = chunk_data.get('text', '')
            prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
            
            # 使用原始LLM生成
            response = self.llm.generate(prompt, system_prompt)
            
            # 解析响应
            try:
                json_data = extract_json_from_response(response)
                if json_data and isinstance(json_data, list):
                    processing_time = time.time() - start_time
                    
                    # 记录成功的原始LLM任务性能
                    record_task_performance(
                        task_type="original_llm",
                        duration=processing_time,
                        success=True,
                        chunk_size=len(text)
                    )
                    
                    return {
                        'chunk_id': chunk_data.get('chunk_id', 'unknown'),
                        'atomic_notes': json_data,
                        'source_text': text,
                        'metadata': chunk_data.get('metadata', {})
                    }
            except Exception as parse_error:
                logger.error(f"Original LLM JSON parsing failed: {parse_error}")
            
            # 如果解析失败，返回空结果
            processing_time = time.time() - start_time
            
            # 记录失败的原始LLM任务性能
            record_task_performance(
                task_type="original_llm",
                duration=processing_time,
                success=False,
                chunk_size=len(text)
            )
            
            return self._create_empty_atomic_note(chunk_data)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Original LLM fallback failed: {e}")
            
            # 记录失败的原始LLM任务性能
            record_task_performance(
                task_type="original_llm",
                duration=processing_time,
                success=False,
                chunk_size=len(chunk_data.get('text', ''))
            )
            
            return {
                'chunk_id': chunk_data.get('chunk_id', 'unknown'),
                'atomic_notes': [],
                'source_text': chunk_data.get('text', ''),
                'metadata': chunk_data.get('metadata', {}),
                'error': 'all_methods_failed'
            }
    
    def _log_performance_stats(self):
        """记录性能统计信息"""
        with self._stats_lock:
            total_time = self.stats['ollama_total_time'] + self.stats['lmstudio_total_time']
            total_tasks = self.stats['ollama_tasks'] + self.stats['lmstudio_tasks']
            
            if total_tasks > 0:
                avg_time = total_time / total_tasks
                success_rate = (self.stats['successful_chunks'] / self.stats['total_chunks']) * 100 if self.stats['total_chunks'] > 0 else 0
                
                logger.info(f"Parallel processing stats:")
                logger.info(f"  Total tasks: {total_tasks}")
                logger.info(f"  Ollama tasks: {self.stats['ollama_tasks']}")
                logger.info(f"  LM Studio tasks: {self.stats['lmstudio_tasks']}")
                logger.info(f"  Average time per task: {avg_time:.2f}s")
                logger.info(f"  Success rate: {success_rate:.1f}%")
                logger.info(f"  Fallback count: {self.stats['fallback_count']}")
                logger.info(f"  Parse failures: {self.stats['parse_failures']}")
                logger.info(f"  Retry attempts: {self.stats['retry_attempts']}")
    
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
                'ollama_total_time': 0.0,
                'lmstudio_total_time': 0.0,
                'ollama_errors': 0,
                'lmstudio_errors': 0,
                'fallback_count': 0,
                'total_chunks': 0,
                'successful_chunks': 0,
                'parse_failures': 0,
                'retry_attempts': 0,
                'empty_results': 0
            }
    
    def _handle_performance_alert(self, alert_type: str, message: str, metrics: Dict[str, Any]):
        """处理性能告警"""
        logger.warning(f"Performance Alert [{alert_type}]: {message}")
        logger.info(f"Current metrics: {metrics}")
        
        # 根据告警类型采取相应措施
        if alert_type == "high_error_rate":
            logger.warning("High error rate detected, consider checking service health")
        elif alert_type == "high_avg_duration":
            logger.warning("High average duration detected, consider optimizing or scaling")
        elif alert_type == "high_fallback_rate":
            logger.warning("High fallback rate detected, primary services may be unstable")
        elif alert_type == "consecutive_failures":
            logger.error("Consecutive failures detected, services may be down")
    
    def get_monitor_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return get_monitor().get_metrics() if hasattr(self, 'monitor') else {}