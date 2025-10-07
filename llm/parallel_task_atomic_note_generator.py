from typing import List, Dict, Any, Union, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from .atomic_note_generator import AtomicNoteGenerator
from .local_llm import LocalLLM
from .ollama_client import OllamaClient
from .lmstudio_client import LMStudioClient
from utils.json_utils import extract_json_from_response
from utils.notes_parser import enrich_note_keys, normalize_note_fields, parse_notes_response
from config import config

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

    def _normalize_to_notes(self, note_data):
        """dict -> [dict], list[dict] -> list[dict], "~"/None/"" -> []"""
        if note_data is None:
            return []
        if isinstance(note_data, str):
            s = note_data.strip()
            if s in ("", "~"):
                return []
        if isinstance(note_data, dict):
            return [note_data]
        if isinstance(note_data, list):
            return [x for x in note_data if isinstance(x, dict)]
        return []

    def _batch_convert(self, note_data, chunk_data):
        notes_raw = self._normalize_to_notes(note_data)
        normalized = [normalize_note_fields(n) for n in notes_raw]
        enriched = [enrich_note_keys(n) for n in normalized]
        return [self._convert_to_atomic_note_format(n, chunk_data) for n in enriched]
    
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
        all_notes = []

        # 任务分配
        ollama_tasks, lmstudio_tasks = self._allocate_tasks(text_chunks)

        logger.info(f"Task allocation: Ollama={len(ollama_tasks)}, LM Studio={len(lmstudio_tasks)}")

        # 并行处理 - 减少并发数量以避免Ollama过载
        max_workers = min(4, len(text_chunks))  # 最多4个并发任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            future_to_meta = {}

            # 提交Ollama任务
            for chunk_data, index in ollama_tasks:
                future = executor.submit(self._process_with_ollama, chunk_data, index, system_prompt)
                futures.append(future)
                future_to_meta[future] = (index, 'ollama', chunk_data)

            # 提交LM Studio任务
            for chunk_data, index in lmstudio_tasks:
                future = executor.submit(self._process_with_lmstudio, chunk_data, index, system_prompt)
                futures.append(future)
                future_to_meta[future] = (index, 'lmstudio', chunk_data)

            # 收集结果
            timeout_value = self.fallback_timeout if self.enable_fallback else None
            for future in as_completed(futures):
                index, client_type, chunk_data = future_to_meta.get(future, (-1, 'unknown', {}))
                try:
                    # 使用更长的超时时间，并添加详细的超时信息
                    logger.debug(f"Waiting for {client_type} task {index} with timeout {timeout_value}s")

                    batch = future.result(timeout=timeout_value)
                    normalized_batch: List[Dict[str, Any]] = []
                    if isinstance(batch, list):
                        normalized_batch = [b for b in batch if isinstance(b, dict)]
                    elif isinstance(batch, dict):
                        normalized_batch = [batch]
                    elif batch is not None:
                        logger.warning(
                            f"Unexpected batch type from {client_type} task {index}: {type(batch)}"
                        )

                    if normalized_batch:
                        all_notes.extend(normalized_batch)

                    with self._stats_lock:
                        if client_type == 'ollama':
                            self.stats['ollama_success'] += 1
                        else:
                            self.stats['lmstudio_success'] += 1

                    if progress_tracker:
                        progress_tracker.update(1)

                except Exception as e:
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
                            fallback_result = self._fallback_process(chunk_data, system_prompt, client_type)
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed for index {index}: {fallback_error}")
                            fallback_result = []

                        normalized_fallback: List[Dict[str, Any]] = []
                        if isinstance(fallback_result, list):
                            normalized_fallback = [item for item in fallback_result if isinstance(item, dict)]
                        elif isinstance(fallback_result, dict):
                            normalized_fallback = [fallback_result]
                        elif fallback_result is not None:
                            logger.warning(
                                f"Unexpected fallback type for index {index}: {type(fallback_result)}"
                            )

                        if not normalized_fallback:
                            normalized_fallback = [self._create_fallback_note(chunk_data)]

                        all_notes.extend(normalized_fallback)
                        with self._stats_lock:
                            self.stats['fallback_count'] += 1
                    else:
                        all_notes.append(self._create_fallback_note(chunk_data))
                    if progress_tracker:
                        progress_tracker.update(1)

        # 后处理
        for i, note in enumerate(all_notes):
            if note is None:
                all_notes[i] = self._create_fallback_note({'text': ''})

            if not isinstance(all_notes[i], dict):
                all_notes[i] = {'content': str(note), 'error': True}

            all_notes[i]['note_id'] = f"note_{i:06d}"
            all_notes[i]['created_at'] = self._get_timestamp()

        # 记录统计信息
        if self.monitoring_enabled:
            self._log_performance_stats()

        logger.info(f"Parallel task division completed: {len(all_notes)} notes generated")
        return all_notes
    
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
    
    def _process_with_ollama(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """使用Ollama处理单个任务"""
        start_time = time.time()
        try:
            text = chunk_data.get('text', '')
            prompt = self._format_atomic_note_prompt(text)
            
            logger.debug(f"Ollama processing task {index}, text length: {len(text)}")
            
            # 添加超时控制
            response = self.ollama_client.generate(prompt, system_prompt, timeout=30)
            
            # 检查响应是否为空
            if not response or response.strip() == "":
                logger.error(f"Ollama task {index}: Empty response received")
                raise ValueError("Ollama returned empty response - possible connection or model issue")
            
            logger.debug(f"Ollama task {index}: Response length: {len(response)}")
            logger.debug(f"Ollama task {index}: Raw response: {response[:500]}...")
            
            # 解析响应
            cleaned_response = extract_json_from_response(response)
            parse_target = cleaned_response or response

            logger.debug(
                f"Ollama task {index}: Cleaned JSON candidate length: {len(parse_target)}"
            )

            parsed_notes = parse_notes_response(parse_target)
            if parsed_notes is None:
                logger.error(
                    f"Ollama task {index}: No valid JSON in response: {response[:200]}..."
                )
                raise ValueError(
                    f"No valid JSON found in Ollama response: {response[:200]}..."
                )

            results = self._batch_convert(parsed_notes, chunk_data)

            # Debug log the generated content
            if results:
                first_note = results[0]
                content = first_note.get('content') if isinstance(first_note, dict) else None
                content_length = len(content) if content else 0
                logger.debug(f"Ollama task {index}: Generated {len(results)} notes, first content length: {content_length}, preview: {content[:100] if content else 'EMPTY'}")
            else:
                logger.debug(f"Ollama task {index}: No valid notes produced from response")
            
            # 记录处理时间
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['ollama_total_time'] += processing_time
            
            logger.debug(f"Ollama task {index}: Completed in {processing_time:.2f}s")
            return results
            
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
    
    def _process_with_lmstudio(self, chunk_data: Dict[str, Any], index: int, system_prompt: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """使用LM Studio处理单个任务"""
        start_time = time.time()
        try:
            text = chunk_data.get('text', '')
            prompt = self._format_atomic_note_prompt(text)
            
            logger.debug(f"LMStudio processing task {index}, text length: {len(text)}")
            
            response = self.lmstudio_client.generate(prompt, system_prompt)
            
            # 检查响应是否为空
            if not response or response.strip() == "":
                logger.error(f"LMStudio task {index}: Empty response received")
                raise ValueError("LMStudio returned empty response - possible connection or model issue")
            
            logger.debug(f"LMStudio task {index}: Response length: {len(response)}")
            logger.debug(f"LMStudio task {index}: Raw response: {response[:500]}...")
            
            # 解析响应
            cleaned_response = extract_json_from_response(response)
            parse_target = cleaned_response or response

            logger.debug(
                f"LMStudio task {index}: Cleaned JSON candidate length: {len(parse_target)}"
            )

            parsed_notes = parse_notes_response(parse_target)
            if parsed_notes is None:
                logger.error(
                    f"LMStudio task {index}: No valid JSON in response: {response[:200]}..."
                )
                raise ValueError(
                    f"No valid JSON found in LM Studio response: {response[:200]}..."
                )

            results = self._batch_convert(parsed_notes, chunk_data)

            # Debug log the generated content
            if results:
                first_note = results[0]
                content = first_note.get('content') if isinstance(first_note, dict) else None
                content_length = len(content) if content else 0
                logger.debug(f"LMStudio task {index}: Generated {len(results)} notes, first content length: {content_length}, preview: {content[:100] if content else 'EMPTY'}")
            else:
                logger.debug(f"LMStudio task {index}: No valid notes produced from response")
            
            # 记录处理时间
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.stats['lmstudio_total_time'] += processing_time
            raise e
    
    def _process_chunk_lmstudio(self, chunk_data: Dict[str, Any], system_prompt: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """使用LM Studio处理单个chunk"""
        try:
            text = chunk_data.get('text', '')
            prompt = self._format_atomic_note_prompt(text)
            
            # 调用LM Studio生成
            response = self.lmstudio_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1024
            )
            
            raw_response = response.get('content', '')
            logger.debug(f"LM Studio raw response for chunk: {raw_response[:200]}...")

            cleaned_response = extract_json_from_response(raw_response)

            # 解析响应
            parsed_notes = parse_notes_response(cleaned_response or raw_response)
            parsed_count = len(parsed_notes) if isinstance(parsed_notes, list) else 0
            logger.debug(f"Parsed {parsed_count} notes from LM Studio response")

            if not parsed_notes:
                logger.warning("No notes parsed from LM Studio response, creating fallback")
                return self._create_fallback_note(chunk_data)

            # 转换为原子笔记格式
            results = self._batch_convert(parsed_notes, chunk_data)
            if not results:
                logger.warning("Parsed notes could not be converted, creating fallback")
                return self._create_fallback_note(chunk_data)

            first_note = results[0]
            content = first_note.get('content') if isinstance(first_note, dict) else ''
            logger.debug(f"Generated {len(results)} atomic notes, first content length: {len(content)}")

            return results
            
        except Exception as e:
            logger.error(f"LM Studio processing failed: {e}")
            return self._create_fallback_note(chunk_data)
            
    def _fallback_process(self, chunk_data: Dict[str, Any], system_prompt: str, failed_client: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
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
    
    def _fallback_to_original_llm(self, chunk_data: Dict[str, Any], system_prompt: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """使用原始LLM作为最终回退"""
        try:
            text = chunk_data.get('text', '')
            prompt = self._format_atomic_note_prompt(text)
            
            response = self.llm.generate(prompt, system_prompt)

            # 解析响应
            cleaned_response = extract_json_from_response(response)
            parse_target = cleaned_response or response
            parsed_notes = parse_notes_response(parse_target)
            if parsed_notes is None:
                raise ValueError("No valid JSON found in original LLM response")

            return self._batch_convert(parsed_notes, chunk_data)
            
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
    
    def _create_atomic_note_from_data(self, note_data: Dict[str, Any], chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """从解析的笔记数据创建原子笔记格式
        
        这个方法将解析后的笔记数据转换为标准的原子笔记格式。
        它是 AtomicNoteGenerator 中 _convert_to_atomic_note_format 方法的别名。
        """
        return self._convert_to_atomic_note_format(note_data, chunk_data)
    
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