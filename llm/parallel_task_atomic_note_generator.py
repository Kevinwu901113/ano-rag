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
        
        # 添加循環後援防護機制
        self._fallback_tracking = {}  # 追蹤每個任務的後援歷史
        self._max_fallback_depth = 2  # 最大後援深度，防止無限循環
    
    def _can_fallback(self, task_id: str, from_client: str, to_client: str) -> bool:
        """檢查是否可以進行後援調用，防止循環後援"""
        if task_id not in self._fallback_tracking:
            self._fallback_tracking[task_id] = []
        
        fallback_history = self._fallback_tracking[task_id]
        
        # 檢查後援深度
        if len(fallback_history) >= self._max_fallback_depth:
            logger.warning(f"Task {task_id}: Maximum fallback depth ({self._max_fallback_depth}) reached")
            return False
        
        # 檢查是否會形成循環
        fallback_chain = f"{from_client}->{to_client}"
        if fallback_chain in fallback_history:
            logger.warning(f"Task {task_id}: Circular fallback detected: {fallback_chain}")
            return False
        
        # 記錄這次後援嘗試
        fallback_history.append(fallback_chain)
        return True
        
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
        """初始化Ollama和LM Studio客户端，即便首次不可用也保留在候选池中"""
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
                
                # 测试Ollama连接，但不管结果如何都保留在候选池中
                if self.ollama_client.is_available():
                    logger.info("Ollama client connection verified - high priority")
                else:
                    logger.warning("Ollama client is not available initially, keeping it in candidate pool with low priority for lazy activation")
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
                
                # 测试LM Studio连接 - 即便首次 is_available() False，也登记成候选（低权重）
                if self.lmstudio_client.is_available():
                    logger.info("LM Studio client connection verified - high priority")
                else:
                    logger.warning("LM Studio client is not available initially, keeping it in candidate pool with low priority for lazy activation")
                    # 不设置为None，保留在候选池中以便后续惰性探活提升权重
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
        """使用任务分配的并行处理方式，支持可恢复错误的重试机制"""
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
            
            # 收集结果，支持可恢复错误的重试
            completed_count = 0
            failed_tasks = []  # 记录失败的任务以便重试
            
            for future in as_completed(future_to_info):
                client_type, index, chunk_data = future_to_info[future]
                
                try:
                    result = future.result()
                    
                    # 检查结果是否有效
                    if result and result.get('atomic_notes') and len(result['atomic_notes']) > 0:
                        with self._stats_lock:
                            self.stats['successful_chunks'] += 1
                        results.append((index, result))
                    elif result and result.get('error') == 'parsing_failed':
                        # 解析失败，记录为可重试的失败任务
                        logger.warning(f"Task {index} ({client_type}) parsing failed, marking for retry")
                        failed_tasks.append((index, chunk_data, client_type))
                    else:
                        # 其他类型的失败，直接使用空结果
                        empty_result = self._create_empty_atomic_note(chunk_data)
                        results.append((index, empty_result))
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Task {index} ({client_type}) failed: {e}")
                    
                    # 检查是否为可恢复错误（No valid JSON、超时等）
                    if self._is_recoverable_error(error_msg):
                        logger.warning(f"Task {index} ({client_type}) has recoverable error, marking for retry")
                        failed_tasks.append((index, chunk_data, client_type))
                    else:
                        # 不可恢复错误，直接创建空结果
                        empty_result = self._create_empty_atomic_note(chunk_data)
                        results.append((index, empty_result))
                
                completed_count += 1
                if progress_tracker:
                    progress_tracker.update(completed_count / len(text_chunks))
            
            # 处理失败的任务：重试一次，仍失败再按 enable_fallback 切到另一端
            if failed_tasks:
                logger.info(f"Retrying {len(failed_tasks)} failed tasks")
                enable_fallback = self.parallel_config.get('enable_fallback', True)
                
                for index, chunk_data, failed_client in failed_tasks:
                    try:
                        # 重试一次
                        if failed_client == 'ollama' and self.ollama_client:
                            result = self._process_with_ollama(chunk_data, index, system_prompt)
                        elif failed_client == 'lmstudio' and self.lmstudio_client:
                            result = self._process_with_lmstudio(chunk_data, index, system_prompt)
                        else:
                            result = None
                        
                        # 检查重试结果
                        if result and result.get('atomic_notes') and len(result['atomic_notes']) > 0:
                            logger.info(f"Task {index} retry succeeded")
                            results.append((index, result))
                            continue
                        
                        # 重试仍失败，按 enable_fallback 决定是否切到另一端
                        if enable_fallback:
                            logger.warning(f"Task {index} retry failed, attempting fallback to other client")
                            if failed_client == 'ollama' and self.lmstudio_client:
                                fallback_result = self._process_with_lmstudio(chunk_data, index, system_prompt)
                            elif failed_client == 'lmstudio' and self.ollama_client:
                                fallback_result = self._process_with_ollama(chunk_data, index, system_prompt)
                            else:
                                fallback_result = self._create_empty_atomic_note(chunk_data)
                            
                            results.append((index, fallback_result))
                        else:
                            # 不启用回退，直接使用空结果
                            empty_result = self._create_empty_atomic_note(chunk_data)
                            results.append((index, empty_result))
                            
                    except Exception as e:
                        logger.error(f"Task {index} retry also failed: {e}")
                        empty_result = self._create_empty_atomic_note(chunk_data)
                        results.append((index, empty_result))
        
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
        
        # 调试日志：检查客户端状态 - 使用惰性探活策略
        ollama_available = self.ollama_client is not None
        lmstudio_available = self.lmstudio_client is not None
        
        logger.info(f"Client status - Ollama: {'Initialized' if ollama_available else 'None'}, "
                   f"LMStudio: {'Initialized' if lmstudio_available else 'None'}")
        
        if allocation_strategy == 'round_robin':
            # 轮询分配 - 使用惰性探活，即使初始不可用也分配少量任务
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if i % 2 == 0:
                    # 偶数索引优先分配给Ollama
                    if ollama_available:
                        ollama_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} assigned to Ollama")
                    elif lmstudio_available:
                        lmstudio_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} fallback to LMStudio (Ollama unavailable)")
                    else:
                        logger.warning(f"Task {i} cannot be assigned - no clients available")
                else:
                    # 奇数索引优先分配给LMStudio
                    if lmstudio_available:
                        lmstudio_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} assigned to LMStudio")
                    elif ollama_available:
                        ollama_tasks.append((chunk_with_id, i))
                        logger.debug(f"Task {i} fallback to Ollama (LMStudio unavailable)")
                    else:
                        logger.warning(f"Task {i} cannot be assigned - no clients available")
        
        elif allocation_strategy == 'ollama_first':
            # 优先使用Ollama
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if ollama_available:
                    ollama_tasks.append((chunk_with_id, i))
                elif lmstudio_available:
                    lmstudio_tasks.append((chunk_with_id, i))
        
        elif allocation_strategy == 'lmstudio_first':
            # 优先使用LM Studio
            for i, chunk in enumerate(text_chunks):
                chunk_with_id = {**chunk, 'chunk_id': chunk.get('chunk_id', f'chunk_{i}')}
                if lmstudio_available:
                    lmstudio_tasks.append((chunk_with_id, i))
                elif ollama_available:
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
            # 使用 replace 方法避免 text 中的花括号导致 format 错误
            prompt = ATOMIC_NOTEGEN_PROMPT.replace('{text}', text)
            
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
            
            # 兩次嘗試都失敗，檢查是否可以進行後援調用
            logger.warning(f"Ollama task {index} retry also failed: {error_msg}")
            
            # 檢查循環後援防護
            if self._can_fallback(task_id, "ollama", "lmstudio"):
                logger.info(f"Attempting fallback from Ollama to LM Studio for task {index}")
                with self._stats_lock:
                    self.stats['fallback_count'] += 1
                
                # 記錄後援操作
                record_fallback_operation(
                    from_client="ollama",
                    to_client="lmstudio",
                    success=False
                )
                
                if self.lmstudio_client:
                    return self._process_with_lmstudio(chunk_data, index, system_prompt)
            else:
                logger.warning(f"Fallback blocked for task {index} to prevent circular calls")
            
            # 沒有LM Studio可用或後援被阻止，返回空結果
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
            # 使用 replace 方法避免 text 中的花括号导致 format 错误
            prompt = ATOMIC_NOTEGEN_PROMPT.replace('{text}', text)
            
            logger.debug(f"LM Studio processing task {index}, text length: {len(text)}")
            
            # 检查客户端是否可用（惰性探活）
            if not self.lmstudio_client.is_available():
                logger.warning(f"LM Studio client not available for task {index}, attempting fallback")
                return self._fallback_process(chunk_data, system_prompt, "lmstudio")
            
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
            
            # 第一次解析失败，检查是否为可恢复错误
            if self._is_recoverable_error(error_msg):
                logger.warning(f"LM Studio recoverable error: {error_msg}, retrying once")
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
            
            # 解析失敗或不可恢復錯誤，檢查是否可以進行後援調用
            logger.warning(f"LM Studio processing failed: {error_msg}")
            with self._stats_lock:
                self.stats['parse_failures'] += 1
            
            # 檢查循環後援防護
            if self._can_fallback(task_id, "lmstudio", "ollama"):
                logger.info(f"Attempting fallback from LM Studio to Ollama for task {index}")
                with self._stats_lock:
                    self.stats['fallback_count'] += 1
                
                # 記錄後援操作
                record_fallback_operation(
                    from_client="lmstudio",
                    to_client="ollama",
                    success=False
                )
                
                if self.ollama_client:
                    return self._process_with_ollama(chunk_data, index, system_prompt)
            else:
                logger.warning(f"Fallback blocked for task {index} to prevent circular calls")
            
            # 沒有Ollama可用或後援被阻止，返回空結果
            logger.error(f"Both LM Studio and Ollama failed for task {index}, returning empty result")
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
            
            # 对于可恢复的错误，尝试回退；否则返回空结果
            if self._is_recoverable_error(error_msg):
                return self._fallback_process(chunk_data, system_prompt, "lmstudio")
            else:
                return self._create_empty_atomic_note(chunk_data)
    
    def _is_recoverable_error(self, error_msg: str) -> bool:
        """判断错误是否可恢复（可以重试或回退）"""
        recoverable_patterns = [
            "no valid json found",
            "json parsing",
            "parsing_failed",
            "invalid json",
            "json decode error",
            "timeout",  # 超时错误通常可以重试
            "connection reset",  # 连接重置可以重试
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in recoverable_patterns)
    
    def _fallback_process(self, chunk_data: Dict[str, Any], system_prompt: str, failed_client: str) -> Dict[str, Any]:
        """回退处理机制，包含重试逻辑和循環防護"""
        start_time = time.time()
        task_id = f"fallback_task_{hash(str(chunk_data))}"
        logger.info(f"Attempting fallback for failed {failed_client} task")
        
        try:
            # 檢查循環後援防護並嘗試回退
            if failed_client == 'ollama' and self.lmstudio_client:
                if self._can_fallback(task_id, "ollama", "lmstudio"):
                    result = self._process_with_lmstudio(chunk_data, -1, system_prompt)
                    
                    # 记录成功的回退操作
                    record_fallback_operation(
                        from_client="ollama",
                        to_client="lmstudio",
                        success=True
                    )
                    
                    return result
                else:
                    logger.warning("Fallback from Ollama to LM Studio blocked to prevent circular calls")
                
            elif failed_client == 'lmstudio' and self.ollama_client:
                if self._can_fallback(task_id, "lmstudio", "ollama"):
                    result = self._process_with_ollama(chunk_data, -1, system_prompt)
                    
                    # 记录成功的回退操作
                    record_fallback_operation(
                        from_client="lmstudio",
                        to_client="ollama",
                        success=True
                    )
                    
                    return result
                else:
                    logger.warning("Fallback from LM Studio to Ollama blocked to prevent circular calls")
            
            # 如果没有其他客户端可用或被阻止，回退到原始LLM
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
            # 使用 replace 方法避免 text 中的花括号导致 format 错误
            prompt = ATOMIC_NOTEGEN_PROMPT.replace('{text}', text)
            
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