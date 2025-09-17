from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
import time
import json
from llm.ollama_client import OllamaClient
from llm.lmstudio_client import LMStudioClient
from utils.json_utils import extract_json_from_response
from config import config
from llm.prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)

class ParallelAtomicNoteGenerator:
    """并行原子笔记生成器，同时调用Ollama qwen2.5和LM Studio qwen2.5提高效率"""
    
    def __init__(self):
        # 获取配置
        self.config = config.get('atomic_note_generation', {})
        self.parallel_strategy = self.config.get('parallel_strategy', 'fastest_wins')
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        self.max_concurrent_chunks = self.config.get('max_concurrent_chunks', 4)
        self.batch_size = self.config.get('batch_size', 8)
        
        # 初始化客户端
        self.ollama_client = None
        self.lmstudio_client = None
        self._init_clients()
        
        # 性能统计
        self.stats = {
            'ollama_wins': 0,
            'lmstudio_wins': 0,
            'ollama_total_time': 0.0,
            'lmstudio_total_time': 0.0,
            'ollama_calls': 0,
            'lmstudio_calls': 0,
            'quality_selections': 0,
            'dual_validations': 0,
            'work_divisions': 0,
            'ollama_assigned': 0,
            'lmstudio_assigned': 0,
            'fallback_switches': 0
        }
        self._lock = threading.Lock()
    
    def _init_clients(self):
        """初始化Ollama和LM Studio客户端"""
        try:
            # 初始化Ollama客户端
            ollama_model = self.config.get('ollama_model', 'qwen2.5:latest')
            self.ollama_client = OllamaClient(model=ollama_model)
            logger.info(f"Initialized Ollama client with model: {ollama_model}")
            
            # 初始化LM Studio客户端
            lmstudio_model = self.config.get('lmstudio_model', 'qwen2.5-7b-instruct')
            lmstudio_port = config.get('llm.multi_model_instances', [{}])[0].get('port', 1234)
            self.lmstudio_client = LMStudioClient(
                model=lmstudio_model,
                port=lmstudio_port
            )
            logger.info(f"Initialized LM Studio client with model: {lmstudio_model} on port: {lmstudio_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize parallel clients: {e}")
            raise
    
    def generate_parallel(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """并行生成原子笔记"""
        if not text_chunks:
            return []
        
        logger.info(f"Starting parallel atomic note generation for {len(text_chunks)} chunks using {self.parallel_strategy} strategy")
        
        results = []
        
        # 分批处理以控制并发数
        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i:i + self.batch_size]
            batch_index = i // self.batch_size  # 计算当前批次的索引
            batch_results = self._process_batch_parallel(batch, progress_tracker, batch_index)
            results.extend(batch_results)
            
            if progress_tracker:
                progress_tracker.update(len(batch_results))
        
        self._log_performance_stats()
        return results
    
    def _process_batch_parallel(self, batch: List[Dict[str, Any]], progress_tracker: Optional[Any] = None, batch_index: int = 0) -> List[Dict[str, Any]]:
        """并行处理一批文本块"""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent_chunks, len(batch))) as executor:
            # 提交所有任务
            future_to_chunk = {}
            for i, chunk_data in enumerate(batch):
                # 为work_division策略计算chunk的索引
                chunk_index = batch_index * len(batch) + i
                future = executor.submit(self._generate_single_note_parallel, chunk_data, chunk_index)
                future_to_chunk[future] = chunk_data
            
            # 收集结果
            for future in as_completed(future_to_chunk):
                chunk_data = future_to_chunk[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to generate note for chunk: {e}")
                    # 创建fallback note
                    fallback_note = self._create_fallback_note(chunk_data)
                    results.append(fallback_note)
        
        return results
    
    def _generate_single_note_parallel(self, chunk_data: Dict[str, Any], batch_index: int = 0) -> Dict[str, Any]:
        """为单个文本块并行生成原子笔记"""
        text = chunk_data.get('text', '')
        # 使用 replace 方法避免 text 中的花括号导致 format 错误
        prompt = ATOMIC_NOTEGEN_PROMPT.replace('{text}', text)
        system_prompt = ATOMIC_NOTEGEN_SYSTEM_PROMPT
        
        if self.parallel_strategy == 'fastest_wins':
            return self._fastest_wins_strategy(prompt, system_prompt, chunk_data)
        elif self.parallel_strategy == 'quality_selection':
            return self._quality_selection_strategy(prompt, system_prompt, chunk_data)
        elif self.parallel_strategy == 'dual_validation':
            return self._dual_validation_strategy(prompt, system_prompt, chunk_data)
        elif self.parallel_strategy == 'work_division':
            return self._work_division_strategy(prompt, system_prompt, chunk_data, batch_index)
        else:
            logger.warning(f"Unknown strategy {self.parallel_strategy}, falling back to fastest_wins")
            return self._fastest_wins_strategy(prompt, system_prompt, chunk_data)
    
    def _fastest_wins_strategy(self, prompt: str, system_prompt: str, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """最快响应策略：返回最先完成的结果"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 同时提交到两个客户端
            ollama_future = executor.submit(self._call_ollama, prompt, system_prompt)
            lmstudio_future = executor.submit(self._call_lmstudio, prompt, system_prompt)
            
            # 等待第一个完成的结果
            for future in as_completed([ollama_future, lmstudio_future], timeout=self.timeout_seconds):
                try:
                    response, client_type, duration = future.result()
                    if response:
                        with self._lock:
                            if client_type == 'ollama':
                                self.stats['ollama_wins'] += 1
                            else:
                                self.stats['lmstudio_wins'] += 1
                        
                        return self._parse_response_to_note(response, chunk_data)
                except Exception as e:
                    logger.warning(f"Client failed: {e}")
                    continue
            
            # 如果都失败了，返回fallback
            return self._create_fallback_note(chunk_data)
    
    def _quality_selection_strategy(self, prompt: str, system_prompt: str, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """质量选择策略：等待两个结果，选择质量更好的"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            ollama_future = executor.submit(self._call_ollama, prompt, system_prompt)
            lmstudio_future = executor.submit(self._call_lmstudio, prompt, system_prompt)
            
            ollama_result = None
            lmstudio_result = None
            
            # 收集两个结果
            for future in as_completed([ollama_future, lmstudio_future], timeout=self.timeout_seconds):
                try:
                    response, client_type, duration = future.result()
                    if client_type == 'ollama':
                        ollama_result = response
                    else:
                        lmstudio_result = response
                except Exception as e:
                    logger.warning(f"Client {future} failed: {e}")
            
            # 选择质量更好的结果
            selected_response = self._select_better_quality(ollama_result, lmstudio_result)
            
            with self._lock:
                self.stats['quality_selections'] += 1
            
            if selected_response:
                return self._parse_response_to_note(selected_response, chunk_data)
            else:
                return self._create_fallback_note(chunk_data)
    
    def _dual_validation_strategy(self, prompt: str, system_prompt: str, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """双重验证策略：两个结果必须一致才接受"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            ollama_future = executor.submit(self._call_ollama, prompt, system_prompt)
            lmstudio_future = executor.submit(self._call_lmstudio, prompt, system_prompt)
            
            ollama_result = None
            lmstudio_result = None
            
            # 收集两个结果
            for future in as_completed([ollama_future, lmstudio_future], timeout=self.timeout_seconds):
                try:
                    response, client_type, duration = future.result()
                    if client_type == 'ollama':
                        ollama_result = response
                    else:
                        lmstudio_result = response
                except Exception as e:
                    logger.warning(f"Client {future} failed: {e}")
            
            # 验证一致性
            if self._validate_consistency(ollama_result, lmstudio_result):
                with self._lock:
                    self.stats['dual_validations'] += 1
                return self._parse_response_to_note(ollama_result or lmstudio_result, chunk_data)
            else:
                # 不一致时选择质量更好的
                selected_response = self._select_better_quality(ollama_result, lmstudio_result)
                if selected_response:
                    return self._parse_response_to_note(selected_response, chunk_data)
                else:
                    return self._create_fallback_note(chunk_data)
    
    def _work_division_strategy(self, prompt: str, system_prompt: str, chunk_data: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
        """工作分工策略：根据batch_index分配任务给不同的模型"""
        # 获取工作分工配置
        work_division_config = config.get('atomic_note_generation', {}).get('work_division', {})
        enable_fallback = work_division_config.get('enable_fallback', True)
        fallback_timeout = work_division_config.get('fallback_timeout', 10)
        
        # 根据batch_index分配模型：偶数给Ollama，奇数给LM Studio
        use_ollama = (batch_index % 2 == 0)
        
        try:
            if use_ollama:
                # 使用Ollama
                with self._lock:
                    self.stats['work_divisions'] += 1
                    self.stats['ollama_assigned'] += 1
                
                response, model_name, duration = self._call_ollama(prompt, system_prompt)
                if response:
                    return self._parse_response_to_note(response, chunk_data)
                elif enable_fallback:
                    # Ollama失败，回退到LM Studio
                    logger.warning(f"Ollama failed for batch {batch_index}, falling back to LM Studio")
                    with self._lock:
                        self.stats['fallback_switches'] += 1
                        self.stats['lmstudio_assigned'] += 1
                    
                    response, model_name, duration = self._call_lmstudio(prompt, system_prompt)
                    if response:
                        return self._parse_response_to_note(response, chunk_data)
            else:
                # 使用LM Studio
                with self._lock:
                    self.stats['work_divisions'] += 1
                    self.stats['lmstudio_assigned'] += 1
                
                response, model_name, duration = self._call_lmstudio(prompt, system_prompt)
                if response:
                    return self._parse_response_to_note(response, chunk_data)
                elif enable_fallback:
                    # LM Studio失败，回退到Ollama
                    logger.warning(f"LM Studio failed for batch {batch_index}, falling back to Ollama")
                    with self._lock:
                        self.stats['fallback_switches'] += 1
                        self.stats['ollama_assigned'] += 1
                    
                    response, model_name, duration = self._call_ollama(prompt, system_prompt)
                    if response:
                        return self._parse_response_to_note(response, chunk_data)
        
        except Exception as e:
            logger.error(f"Work division strategy failed for batch {batch_index}: {e}")
        
        # 所有尝试都失败，返回fallback note
        return self._create_fallback_note(chunk_data)
    
    def _call_ollama(self, prompt: str, system_prompt: str) -> Tuple[str, str, float]:
        """调用Ollama客户端"""
        start_time = time.time()
        try:
            response = self.ollama_client.generate(prompt, system_prompt)
            duration = time.time() - start_time
            
            with self._lock:
                self.stats['ollama_calls'] += 1
                self.stats['ollama_total_time'] += duration
            
            return response, 'ollama', duration
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    def _call_lmstudio(self, prompt: str, system_prompt: str) -> Tuple[str, str, float]:
        """调用LM Studio客户端"""
        start_time = time.time()
        try:
            response = self.lmstudio_client.generate(prompt, system_prompt)
            duration = time.time() - start_time
            
            with self._lock:
                self.stats['lmstudio_calls'] += 1
                self.stats['lmstudio_total_time'] += duration
            
            return response, 'lmstudio', duration
        except Exception as e:
            logger.error(f"LM Studio call failed: {e}")
            raise
    
    def _select_better_quality(self, ollama_result: str, lmstudio_result: str) -> str:
        """选择质量更好的结果"""
        if not ollama_result and not lmstudio_result:
            return None
        if not ollama_result:
            return lmstudio_result
        if not lmstudio_result:
            return ollama_result
        
        # 简单的质量评估：JSON完整性、内容长度、实体数量等
        ollama_score = self._calculate_quality_score(ollama_result)
        lmstudio_score = self._calculate_quality_score(lmstudio_result)
        
        return ollama_result if ollama_score >= lmstudio_score else lmstudio_result
    
    def _calculate_quality_score(self, response: str) -> float:
        """计算响应质量分数"""
        if not response:
            return 0.0
        
        score = 0.0
        
        # JSON完整性检查
        try:
            cleaned_response = extract_json_from_response(response)
            if cleaned_response:
                data = json.loads(cleaned_response)
                score += 10.0  # 基础JSON分数
                
                # 检查必要字段
                if isinstance(data, dict):
                    if 'content' in data and data['content']:
                        score += 5.0
                    if 'entities' in data and data['entities']:
                        score += 3.0 + len(data['entities']) * 0.5
                    if 'relations' in data and data['relations']:
                        score += 2.0 + len(data['relations']) * 0.3
        except:
            pass
        
        # 内容长度评估
        score += min(len(response) / 100, 5.0)
        
        return score
    
    def _validate_consistency(self, ollama_result: str, lmstudio_result: str) -> bool:
        """验证两个结果的一致性"""
        if not ollama_result or not lmstudio_result:
            return False
        
        try:
            ollama_data = json.loads(extract_json_from_response(ollama_result) or '{}')
            lmstudio_data = json.loads(extract_json_from_response(lmstudio_result) or '{}')
            
            # 检查关键字段的一致性
            ollama_entities = set(ollama_data.get('entities', []))
            lmstudio_entities = set(lmstudio_data.get('entities', []))
            
            # 如果实体重叠度超过70%，认为一致
            if ollama_entities and lmstudio_entities:
                overlap = len(ollama_entities & lmstudio_entities)
                total = len(ollama_entities | lmstudio_entities)
                consistency_ratio = overlap / total if total > 0 else 0
                return consistency_ratio >= 0.7
            
            return False
        except:
            return False
    
    def _parse_response_to_note(self, response: str, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """将响应解析为原子笔记格式"""
        try:
            cleaned_response = extract_json_from_response(response)
            if not cleaned_response:
                return self._create_fallback_note(chunk_data)
            
            note_data = json.loads(cleaned_response)
            
            # 处理列表格式
            if isinstance(note_data, list) and note_data:
                note_data = note_data[0]
            
            if not isinstance(note_data, dict):
                return self._create_fallback_note(chunk_data)
            
            # 创建标准格式的原子笔记
            text = chunk_data.get('text', '')
            return {
                'original_text': text,
                'content': note_data.get('content', text),
                'summary': note_data.get('summary', note_data.get('content', text)),
                'title': self._extract_title_from_chunk(chunk_data),
                'raw_span': text,
                'raw_span_evidence': self._generate_raw_span_evidence(
                    note_data.get('entities', []),
                    note_data.get('relations', []),
                    text
                ),
                'entities': self._clean_list(note_data.get('entities', [])),
                'relations': note_data.get('relations', []),
                'paragraph_idx_mapping': chunk_data.get('paragraph_idx_mapping', {}),
                'relevant_paragraph_idxs': self._extract_relevant_paragraph_idxs(
                    text, chunk_data.get('paragraph_idx_mapping', {})
                ),
                'chunk_id': chunk_data.get('chunk_id', ''),
                'source_file': chunk_data.get('source_file', ''),
                'metadata': chunk_data.get('metadata', {})
            }
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return self._create_fallback_note(chunk_data)
    
    def _create_fallback_note(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建fallback原子笔记"""
        text = chunk_data.get('text', '')
        return {
            'original_text': text,
            'content': text,
            'summary': text[:200] + '...' if len(text) > 200 else text,
            'title': self._extract_title_from_chunk(chunk_data),
            'raw_span': text,
            'raw_span_evidence': '',
            'entities': [],
            'relations': [],
            'paragraph_idx_mapping': chunk_data.get('paragraph_idx_mapping', {}),
            'relevant_paragraph_idxs': [],
            'chunk_id': chunk_data.get('chunk_id', ''),
            'source_file': chunk_data.get('source_file', ''),
            'metadata': chunk_data.get('metadata', {})
        }
    
    def _extract_title_from_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """从chunk数据中提取标题"""
        return chunk_data.get('title', chunk_data.get('source_file', 'Untitled'))
    
    def _extract_relevant_paragraph_idxs(self, text: str, paragraph_idx_mapping: Dict[str, Any]) -> List[int]:
        """提取相关段落索引"""
        return list(paragraph_idx_mapping.keys()) if paragraph_idx_mapping else []
    
    def _generate_raw_span_evidence(self, entities: List[str], relations: List[Any], text: str) -> str:
        """生成raw_span_evidence"""
        evidence_parts = []
        if entities:
            evidence_parts.append(f"Entities: {', '.join(entities)}")
        if relations:
            relation_strs = []
            for rel in relations:
                if isinstance(rel, dict):
                    relation_strs.append(f"{rel.get('subject', '')} -> {rel.get('predicate', '')} -> {rel.get('object', '')}")
                else:
                    relation_strs.append(str(rel))
            if relation_strs:
                evidence_parts.append(f"Relations: {'; '.join(relation_strs)}")
        return ' | '.join(evidence_parts)
    
    def _clean_list(self, items: List[Any]) -> List[str]:
        """清理列表，确保所有项目都是字符串"""
        cleaned = []
        for item in items:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif item:
                cleaned.append(str(item).strip())
        return cleaned
    
    def _log_performance_stats(self):
        """记录性能统计信息"""
        with self._lock:
            total_calls = self.stats['ollama_calls'] + self.stats['lmstudio_calls']
            if total_calls > 0:
                avg_ollama_time = self.stats['ollama_total_time'] / max(self.stats['ollama_calls'], 1)
                avg_lmstudio_time = self.stats['lmstudio_total_time'] / max(self.stats['lmstudio_calls'], 1)
                
                logger.info(f"Parallel generation stats:")
                logger.info(f"  Ollama wins: {self.stats['ollama_wins']}, LM Studio wins: {self.stats['lmstudio_wins']}")
                logger.info(f"  Ollama avg time: {avg_ollama_time:.2f}s, LM Studio avg time: {avg_lmstudio_time:.2f}s")
                logger.info(f"  Quality selections: {self.stats['quality_selections']}, Dual validations: {self.stats['dual_validations']}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """重置性能统计"""
        with self._lock:
            self.stats = {
                'ollama_wins': 0,
                'lmstudio_wins': 0,
                'ollama_total_time': 0.0,
                'lmstudio_total_time': 0.0,
                'ollama_calls': 0,
                'lmstudio_calls': 0,
                'quality_selections': 0,
                'dual_validations': 0,
                'work_divisions': 0,
                'ollama_assigned': 0,
                'lmstudio_assigned': 0,
                'fallback_switches': 0
            }