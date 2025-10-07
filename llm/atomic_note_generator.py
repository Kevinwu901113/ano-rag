from typing import List, Dict, Any, Union, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import time
from .local_llm import LocalLLM
from utils.batch_processor import BatchProcessor
from utils.text_utils import TextUtils
from utils.json_utils import extract_json_from_response, clean_control_characters
from utils.notes_parser import (
    enrich_note_keys,
    filter_valid_notes,
    normalize_note_fields,
    parse_notes_response,
)
from utils.notes_quality_filter import NotesQualityFilter
from utils.notes_retry_handler import NotesRetryHandler
from utils.notes_stats_logger import get_global_stats_logger, log_notes_stats, finalize_notes_session
from utils.note_coverage_eval import evaluate_note_coverage
from config import config
from .prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)

class AtomicNoteGenerator:
    """原子笔记生成器，专门用于文档处理阶段的原子笔记构建"""
    
    def __init__(self, llm: LocalLLM = None):
        if llm is None:
            raise ValueError("AtomicNoteGenerator requires a LocalLLM instance to be passed")
        self.llm = llm
        
        # 检查是否为hybrid模式并使用单例HybridLLMDispatcher
        self.is_hybrid_mode = getattr(llm, 'is_hybrid_mode', False)
        self.hybrid_dispatcher = None
        if self.is_hybrid_mode:
            from .multi_model_client import HybridLLMDispatcher
            self.hybrid_dispatcher = HybridLLMDispatcher()  # 单例模式，自动重用实例
            logger.info("AtomicNoteGenerator using HybridLLMDispatcher singleton instance")
        
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        # 摘要校验器将在需要时动态导入，避免循环导入
        self.summary_auditor = None
        
        # 并发处理配置
        self.concurrent_enabled = config.get('document.concurrent_processing.enabled', True)
        self.max_concurrent_workers = config.get('document.concurrent_processing.max_workers', 4)
        self._lock = threading.Lock()
        
        # 检查LLM是否支持并发处理（包括多模型并行）
        self.concurrent_support = self._check_concurrent_support()
        
        # 新增：优化配置
        self.enable_fast_path = config.get('notes_llm.enable_fast_path', True)
        self.sentinel_char = config.get('notes_llm.sentinel_char', '~')
        
        # 统计信息
        self.processing_stats = {
            'notes_zero_count': 0,
            'sentinel_rate': 0.0,
            'parse_fail_rate': 0.0,
            'rule_fallback_rate': 0.0,
            'total_chunks_processed': 0,
            'total_notes_generated': 0
        }

    def _check_concurrent_support(self) -> bool:
        """检查LLM是否支持并发处理（包括多模型并行和单实例队列）"""
        try:
            # 检查是否是MultiModelClient（多模型并行）
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'model_instances'):
                instances = getattr(self.llm.client, 'model_instances', [])
                return len(instances) > 1
            
            # 检查是否是LMStudioClient且启用了并发
            if hasattr(self.llm, 'lmstudio_client'):
                client = self.llm.lmstudio_client
                return getattr(client, 'concurrent_enabled', False)
            
            # 检查是否直接是并发客户端
            if hasattr(self.llm, 'client'):
                client = self.llm.client
                return (getattr(client, 'concurrent_enabled', False) or 
                       hasattr(client, 'model_instances'))
            
            return False
        except Exception as e:
            logger.warning(f"Failed to check concurrent support: {e}")
            return False
    
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """从文本块生成原子笔记"""
        logger.info(f"Generating atomic notes for {len(text_chunks)} text chunks")
        
        # 重置统计信息
        self.reset_processing_stats()
        start_time = time.time()
        
        # 如果启用并发且支持并发处理，使用并发处理
        if self.concurrent_enabled and self.concurrent_support:
            client = getattr(self.llm, 'lmstudio_client', None) or getattr(self.llm, 'client', None)
            instance_count = len(getattr(client, 'model_instances', getattr(client, 'instances', []))) if client else 1
            logger.info(f"Using concurrent processing with {instance_count} instances")
            atomic_notes = self._generate_atomic_notes_concurrent(text_chunks, progress_tracker)
        else:
            logger.info("Using sequential processing")
            atomic_notes = self._generate_atomic_notes_sequential(text_chunks, progress_tracker)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 更新统计信息
        final_stats = self.get_processing_stats()
        final_stats['total_processing_time'] = processing_time
        
        # 记录到全局统计记录器
        log_notes_stats(final_stats)
        
        # 记录会话摘要
        session_summary = finalize_notes_session(processing_time)
        logger.info(f"Notes generation completed: {session_summary}")
        
        coverage_cfg = config.get('evaluation', {}).get('coverage', {})
        if coverage_cfg:
            try:
                evaluate_note_coverage(text_chunks, atomic_notes)
            except Exception as coverage_error:
                logger.warning(f"Coverage evaluation failed: {coverage_error}")

        return atomic_notes
    
    def _generate_atomic_notes_sequential(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """顺序生成原子笔记（原有逻辑）"""
        # 准备提示词模板
        system_prompt = self._get_atomic_note_system_prompt()
        
        def process_batch(batch):
            if not isinstance(batch, list):
                batch = [batch]

            results = []
            for chunk_data in batch:
                try:
                    notes = self._generate_single_atomic_note(chunk_data, system_prompt)
                    if isinstance(notes, list):
                        results.extend(notes)
                    elif notes:
                        results.append(notes)
                except Exception as e:
                    logger.error(f"Failed to generate atomic note: {e}")
                    fallback_note = self._create_fallback_note(chunk_data)
                    if fallback_note:
                        results.append(fallback_note)
                        self.processing_stats['total_notes_generated'] += 1
            return results
        
        atomic_notes = self.batch_processor.process_batches(
            text_chunks,
            process_batch,
            desc="Generating atomic notes",
            progress_tracker=progress_tracker
        )
        
        # 后处理：添加稳定的ID和元数据
        for i, note in enumerate(atomic_notes):
            # 确保note是字典类型
            if not isinstance(note, dict):
                logger.warning(f"Note at index {i} is not a dict, got {type(note)}: {note}")
                # 如果note是列表，尝试提取第一个元素
                if isinstance(note, list) and len(note) > 0 and isinstance(note[0], dict):
                    note = note[0]
                    atomic_notes[i] = note
                else:
                    # 创建一个基本的字典结构
                    note = {'content': str(note), 'error': True}
                    atomic_notes[i] = note
            
            # 生成基于源文档信息的稳定note_id
            note['note_id'] = self._generate_stable_note_id(note, i)
            note['created_at'] = self._get_timestamp()
        
        # 摘要校验：仅在启用时进行
        if config.get('summary_auditor.enabled', False):
            try:
                from utils.summary_auditor import SummaryAuditor
                logger.info("Starting summary audit for generated atomic notes")
                auditor = SummaryAuditor(llm=self.llm)
                atomic_notes = auditor.batch_audit_summaries(atomic_notes)
                logger.info("Summary audit completed")
            except ImportError as e:
                logger.warning(f"Failed to import SummaryAuditor: {e}")
            except Exception as e:
                logger.error(f"Summary audit failed: {e}")
        
        logger.info(f"Generated {len(atomic_notes)} atomic notes")
        return atomic_notes
    
    def _generate_atomic_notes_concurrent(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """并发生成原子笔记，利用多个LM Studio实例"""
        system_prompt = self._get_atomic_note_system_prompt()
        chunk_notes: List[List[Dict[str, Any]]] = [[] for _ in range(len(text_chunks))]
        
        # 计算实际的并发工作线程数
        client = getattr(self.llm, 'lmstudio_client', None) or getattr(self.llm, 'client', None)
        instance_count = len(getattr(client, 'model_instances', getattr(client, 'instances', []))) if client else 1
        max_workers = min(self.max_concurrent_workers, instance_count, len(text_chunks))
        
        logger.info(f"Starting concurrent processing with {max_workers} workers for {len(text_chunks)} chunks")
        
        def process_chunk_with_index(chunk_index_pair):
            """处理单个文本块并返回索引和结果"""
            chunk_data, index = chunk_index_pair
            try:
                notes = self._generate_single_atomic_note(chunk_data, system_prompt)
                return index, notes, None
            except Exception as e:
                logger.error(f"Failed to generate atomic note for chunk {index}: {e}")
                fallback_note = self._create_fallback_note(chunk_data)
                return index, [fallback_note] if fallback_note else [], e
        
        # 创建索引化的任务列表
        indexed_chunks = [(chunk, i) for i, chunk in enumerate(text_chunks)]
        
        # 使用ThreadPoolExecutor进行并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(process_chunk_with_index, chunk_pair): chunk_pair[1] 
                             for chunk_pair in indexed_chunks}
            
            # 收集结果
            completed_count = 0
            error_count = 0
            
            for future in as_completed(future_to_index):
                try:
                    index, notes, error = future.result()
                    chunk_notes[index] = notes if isinstance(notes, list) else ([notes] if notes else [])

                    if error:
                        error_count += 1
                        self.processing_stats['total_notes_generated'] += len(chunk_notes[index])

                    completed_count += 1
                    
                    # 更新进度跟踪器
                    if progress_tracker:
                        progress_tracker.update(1)
                    
                    if completed_count % 10 == 0:  # 每10个任务记录一次进度
                        logger.info(f"Completed {completed_count}/{len(text_chunks)} atomic notes")
                        
                except Exception as e:
                    original_index = future_to_index[future]
                    logger.error(f"Future execution failed for chunk {original_index}: {e}")
                    # 创建fallback note
                    if original_index < len(text_chunks):
                        fallback_note = self._create_fallback_note(text_chunks[original_index])
                        chunk_notes[original_index] = [fallback_note] if fallback_note else []
                        if fallback_note:
                            self.processing_stats['total_notes_generated'] += 1
                    error_count += 1
                    
                    # 更新进度跟踪器（即使出错也要更新）
                    if progress_tracker:
                        progress_tracker.update(1)
        
        total_generated = sum(len(notes) for notes in chunk_notes)
        logger.info(f"Concurrent processing completed: {total_generated} notes generated, {error_count} errors")

        # 后处理：添加ID和元数据
        flattened_notes: List[Dict[str, Any]] = []
        for i, notes in enumerate(chunk_notes):
            for note in notes:
                if not isinstance(note, dict):
                    logger.warning(f"Note at chunk {i} is not a dict, got {type(note)}: {note}")
                    note = {'content': str(note), 'error': True}
                flattened_notes.append(note)

        for i, note in enumerate(flattened_notes):
            note['note_id'] = f"note_{i:06d}"
            note['created_at'] = self._get_timestamp()
        
        # 摘要校验：仅在启用时进行
        if config.get('summary_auditor.enabled', False):
            try:
                from utils.summary_auditor import SummaryAuditor
                logger.info("Starting summary audit for generated atomic notes")
                auditor = SummaryAuditor(llm=self.llm)
                flattened_notes = auditor.audit_atomic_notes(flattened_notes)
            except Exception as e:
                logger.error(f"Summary audit failed: {e}")

        return flattened_notes
    
    def _generate_single_atomic_note(self, chunk_data: Union[Dict[str, Any], Any], system_prompt: str) -> List[Dict[str, Any]]:
        """生成指定文本块对应的原子笔记列表"""
        # 确保 chunk_data 是字典类型
        if not isinstance(chunk_data, dict):
            logger.warning(f"chunk_data is not a dict, got {type(chunk_data)}: {chunk_data}")
            # 如果是字符串，创建基本的字典结构
            if isinstance(chunk_data, str):
                chunk_data = {'text': chunk_data}
            else:
                logger.error(f"Unsupported chunk_data type: {type(chunk_data)}")
                fallback_note = self._create_fallback_note({'text': str(chunk_data)})
                self.processing_stats['total_notes_generated'] += 1
                return [fallback_note] if fallback_note else []
        
        text = chunk_data.get('text', '')
        self.processing_stats['total_chunks_processed'] += 1
        
        # 准备LLM调用参数
        llm_params = self._get_optimized_llm_params()
        prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
        
        # 定义LLM生成函数
        def llm_generate_func(user_prompt: str, sys_prompt: str) -> str:
            if self.is_hybrid_mode and self.hybrid_dispatcher:
                return self.hybrid_dispatcher.process_single(user_prompt, sys_prompt, **llm_params)
            else:
                # 检查是否启用流式处理和早停机制
                stream_early_stop = config.get('notes_llm.stream_early_stop', False)
                if stream_early_stop and hasattr(self.llm, 'generate_stream'):
                    # 使用流式生成并实现真正的早停逻辑
                    response_parts = []
                    seen_first = False
                    try:
                        for chunk in self.llm.generate_stream(user_prompt, sys_prompt, **llm_params):
                            if not chunk:
                                continue
                            response_parts.append(chunk)
                            
                            # 检测首个非空白字符
                            if not seen_first:
                                head = "".join(response_parts).lstrip()
                                if not head:
                                    continue
                                seen_first = True
                                c0 = head[0]
                                
                                # 立刻早停：无事实
                                if c0 == config.get('notes_llm.sentinel_char', '~'):
                                    return c0
                                
                                # 检查是否为无效前缀，触发回退
                                if c0 not in ['[', config.get('notes_llm.sentinel_char', '~')] and len(head) > 16:
                                    raise RuntimeError("early-stop: invalid prefix")
                        
                        return "".join(response_parts)
                    except Exception as e:
                        logger.warning(f"Stream generation failed, falling back to regular generation: {e}")
                        return self.llm.generate(user_prompt, sys_prompt, **llm_params)
                else:
                    return self.llm.generate(user_prompt, sys_prompt, **llm_params)
        
        # 定义解析函数
        def parse_func(response: str) -> Optional[List[Dict[str, Any]]]:
            return parse_notes_response(response, sentinel=self.sentinel_char)
        
        # 使用重试处理器
        retry_handler = NotesRetryHandler()
        parsed_notes, retry_metadata = retry_handler.process_chunk_with_retry(
            chunk_data, llm_generate_func, parse_func, system_prompt, prompt
        )

        # 更新统计信息
        if retry_metadata.get('used_fallback', False):
            self.processing_stats['rule_fallback_rate'] += 1

        if parsed_notes is None:
            fallback_note = self._create_fallback_note(chunk_data)
            if fallback_note:
                self.processing_stats['total_notes_generated'] += 1
                return [fallback_note]
            self.processing_stats['notes_zero_count'] += 1
            return []

        if len(parsed_notes) == 0:
            self.processing_stats['notes_zero_count'] += 1
            return []

        # 标准化笔记字段
        normalized_notes = [normalize_note_fields(note) for note in parsed_notes]
        enriched_notes = [enrich_note_keys(note) for note in normalized_notes]

        # 对每条笔记执行后处理
        post_processed_notes = []
        for note in enriched_notes:
            processed_note = self._post_process_llm_note(note, chunk_data)
            if processed_note:
                post_processed_notes.append(processed_note)

        if not post_processed_notes:
            self.processing_stats['notes_zero_count'] += 1
            return []

        # 过滤有效笔记
        valid_notes = filter_valid_notes(post_processed_notes)

        if not valid_notes:
            self.processing_stats['notes_zero_count'] += 1
            return []

        # 质量过滤
        quality_filter = NotesQualityFilter()
        filtered_notes = quality_filter.filter_notes(valid_notes)

        stats_logger = get_global_stats_logger()
        if quality_filter.stats and stats_logger:
            stats_logger.add_quality_filter_stats(quality_filter.stats)

        if not filtered_notes:
            self.processing_stats['notes_zero_count'] += 1
            return []

        # 转换为原有格式
        atomic_notes = [self._convert_to_atomic_note_format(note, chunk_data) for note in filtered_notes]

        self.processing_stats['total_notes_generated'] += len(atomic_notes)
        return atomic_notes

    def _post_process_llm_note(
        self,
        note: Dict[str, Any],
        chunk_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """对LLM生成的笔记进行统一的后处理"""
        if not isinstance(note, dict):
            return None

        processed = note.copy()

        raw_text = processed.get('text', '')
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        text = raw_text.strip()
        if not text:
            return None

        sentences = TextUtils.split_by_sentence(text)
        if sentences:
            text = sentences[0]
        else:
            text = text.split('\n')[0].strip()

        max_chars = config.get('notes_llm.max_note_chars', 200)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()

        processed['text'] = text
        processed['sent_count'] = 1

        try:
            salience = float(processed.get('salience', 0.5))
        except (TypeError, ValueError):
            salience = 0.5
        processed['salience'] = max(0.0, min(1.0, salience))

        list_fields = ['local_spans', 'entities', 'years', 'quality_flags']
        for field in list_fields:
            value = processed.get(field)
            if value is None:
                processed[field] = []
                continue
            if isinstance(value, list):
                processed[field] = [v for v in value if v not in (None, '')]
            elif value == "":
                processed[field] = []
            else:
                processed[field] = [value]

        if not processed.get('quality_flags'):
            processed['quality_flags'] = ['OK']

        return processed

    def _get_optimized_llm_params(self) -> Dict[str, Any]:
        """获取优化的LLM调用参数"""
        params = config.get('notes_llm.llm_params', {})
        return {
            'temperature': params.get('temperature', 0),
            'top_p': params.get('top_p', 0),
            'max_tokens': params.get('max_tokens', 128),
            'stop': params.get('stop', ['\n\n', self.sentinel_char])
        }
    
    def _normalize_to_notes(self, note_data: Any) -> List[Dict]:
        """
        将 LLM/上游解析结果统一归一化为 List[Dict]。
        - dict -> [dict]
        - list[dict] -> list[dict]
        - "~" / "" / None -> []
        - 其余类型 -> []
        """
        if note_data is None:
            return []
        if isinstance(note_data, str):
            s = note_data.strip()
            if s == "" or s == "~":
                return []
        if isinstance(note_data, dict):
            return [note_data]
        if isinstance(note_data, list):
            return [x for x in note_data if isinstance(x, dict)]
        return []
    
    def _convert_to_atomic_note_format(self, note: Dict[str, Any], chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """将新格式的笔记转换为原有的原子笔记格式"""
        text = (note.get('text', '') or '').strip()

        # 提取相关的paragraph idx信息
        paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
        explicit_idxs = note.get('paragraph_idxs')
        relevant_idxs: List[int] = []
        if isinstance(explicit_idxs, list) and explicit_idxs:
            for idx in explicit_idxs:
                if isinstance(idx, int):
                    relevant_idxs.append(idx)
                elif isinstance(idx, str):
                    try:
                        relevant_idxs.append(int(idx))
                    except ValueError:
                        logger.debug(f"Invalid paragraph idx value ignored: {idx}")

        if not relevant_idxs:
            base_text = chunk_data.get('text', '') or text  # 优先用chunk原文
            relevant_idxs = self._extract_relevant_paragraph_idxs(base_text, paragraph_idx_mapping)
        
        # 兜底逻辑：当且仅当"每文件一个段落"时，直接赋该段落 idx
        if not relevant_idxs:
            para_info = chunk_data.get('paragraph_info') or []
            if len(para_info) == 1 and isinstance(para_info[0].get('idx', None), int):
                relevant_idxs = [para_info[0]['idx']]
        
        # 调试日志：记录空 paragraph_idxs 的情况
        if not relevant_idxs:
            logger.debug(f"paragraph_idxs empty | file={chunk_data.get('source_info',{}).get('file_name')} "
                         f"| chunk_idx={chunk_data.get('chunk_index')} "
                         f"| note_len={len(text)} | has_mapping={bool(paragraph_idx_mapping)}")
        
        # 提取title和raw_span信息
        title = self._extract_title_from_chunk(chunk_data)
        raw_span = text  # raw_span就是原始文本内容
        
        # 提取实体和关系信息
        entities = note.get('entities', []) or []
        fallback_cfg = (config.get('notes_llm', {}) or {}).get('entities_fallback', {}) or {}
        if (not entities) and fallback_cfg.get('enabled', True) and text:
            try:
                entities = TextUtils.extract_entities_fallback(
                    text,
                    min_len=int(fallback_cfg.get('min_len', 2)),
                    allow_types=fallback_cfg.get('types', ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT'])
                ) or []
            except Exception as err:
                logger.debug(f"entities fallback failed: {err}")
                entities = []

        if entities:
            entities = [str(e).strip() for e in entities if str(e).strip()]
            entities = list(dict.fromkeys(entities))
        relations = []  # 新格式暂不包含relations
        
        # 生成raw_span_evidence
        raw_span_evidence = self._generate_raw_span_evidence(entities, relations, text)
        
        # 构建原子笔记
        atomic_note = {
            'original_text': chunk_data.get('text', ''),
            'content': text,
            'summary': text,  # 保留summary字段用于前端显示
            'title': title,
            'raw_span': raw_span,
            'raw_span_evidence': raw_span_evidence,
            'keywords': [],  # 新格式暂不包含keywords
            'entities': entities,
            'concepts': [],  # 新格式暂不包含concepts
            'relations': relations,
            'normalized_entities': [],
            'normalized_predicates': [],
            'importance_score': note.get('salience', 0.5),
            'note_type': 'fact',
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text),
            'paragraph_idxs': relevant_idxs,
            # 新增字段
            'sent_count': note.get('sent_count', 1),
            'salience': note.get('salience', 0.5),
            'local_spans': note.get('local_spans', []),
            'years': note.get('years', []),
            'quality_flags': note.get('quality_flags', ['OK'])
        }
        
        return atomic_note

    def _generate_raw_span_evidence(self, entities: List[str], relations: List[Any], text: str) -> str:
        """生成raw_span_evidence，由实体与关系抽取器拼接成简单证据句
        
        例如: "A co founded B", "X located in Y"
        """
        evidence_sentences = []
        
        # 从relations字段生成证据句
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, dict):
                    subject = relation.get('subject', '')
                    predicate = relation.get('predicate', relation.get('relation', ''))
                    obj = relation.get('object', relation.get('target', ''))
                    
                    if subject and predicate and obj:
                        # 标准化谓词
                        normalized_predicate = self._normalize_predicate(predicate)
                        evidence_sentence = f"{subject} {normalized_predicate} {obj}"
                        evidence_sentences.append(evidence_sentence)
        
        # 如果没有从relations生成证据句，尝试从实体生成简单的存在性证据
        if not evidence_sentences and entities:
            # 为前几个重要实体生成简单的存在性证据
            for entity in entities[:3]:  # 限制数量
                if entity:
                    evidence_sentences.append(f"{entity} mentioned")
        
        # 如果仍然没有证据句，从文本中提取简单的实体关系
        if not evidence_sentences:
            evidence_sentences = self._extract_simple_relations_from_text(text, entities)
        
        return '; '.join(evidence_sentences) if evidence_sentences else text[:100]  # 回退到文本前100字符
    
    def _normalize_predicate(self, predicate: str) -> str:
        """标准化谓词为常见的关系表达"""
        if not predicate:
            return 'related to'
        
        predicate_lower = predicate.lower().strip()
        
        # 谓词映射表
        predicate_mapping = {
            'founded': 'founded',
            'co-founded': 'co founded',
            'cofounded': 'co founded',
            'established': 'founded',
            'created': 'founded',
            'located': 'located in',
            'based': 'located in',
            'situated': 'located in',
            'member': 'member of',
            'belongs': 'member of',
            'works': 'works for',
            'employed': 'works for',
            'part': 'part of',
            'component': 'part of',
            'instance': 'instance of',
            'type': 'instance of',
            'example': 'instance of'
        }
        
        # 查找匹配的标准谓词
        for key, standard_predicate in predicate_mapping.items():
            if key in predicate_lower:
                return standard_predicate
        
        return predicate_lower
    
    def _extract_simple_relations_from_text(self, text: str, entities: List[str]) -> List[str]:
        """从文本中提取简单的实体关系"""
        import re
        
        evidence_sentences = []
        
        if len(entities) < 2:
            return evidence_sentences
        
        # 简单的关系模式匹配
        relation_patterns = [
            (r'(\w+)\s+(founded|established|created)\s+(\w+)', 'founded'),
            (r'(\w+)\s+(co-?founded)\s+(\w+)', 'co founded'),
            (r'(\w+)\s+(located|based|situated)\s+in\s+(\w+)', 'located in'),
            (r'(\w+)\s+(works?\s+for|employed\s+by)\s+(\w+)', 'works for'),
            (r'(\w+)\s+(member\s+of|belongs\s+to)\s+(\w+)', 'member of'),
            (r'(\w+)\s+(part\s+of|component\s+of)\s+(\w+)', 'part of')
        ]
        
        for pattern, relation in relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    subject = match[0].strip()
                    obj = match[2].strip()
                    
                    # 检查是否是已知实体
                    if subject in entities or obj in entities:
                        evidence_sentences.append(f"{subject} {relation} {obj}")
        
        return evidence_sentences
    
    def _extract_title_from_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """从chunk_data中提取title信息"""
        # 首先尝试从paragraph_info中提取title
        paragraph_info = chunk_data.get('paragraph_info', [])
        if paragraph_info:
            # 查找与当前文本最匹配的段落的title
            text = chunk_data.get('text', '')
            for para_info in paragraph_info:
                if isinstance(para_info, dict):
                    para_text = para_info.get('paragraph_text', '')
                    title = para_info.get('title', '')
                    # 如果段落文本与chunk文本有重叠，使用该段落的title
                    if para_text and text and para_text in text:
                        return title
                    # 或者如果chunk文本在段落文本中
                    elif para_text and text and text in para_text:
                        return title
        
        # 如果没有找到匹配的title，尝试从source_info中提取
        source_info = chunk_data.get('source_info', {})
        if isinstance(source_info, dict):
            title = source_info.get('title', '')
            if title:
                return title
        
        # 最后尝试从文本开头提取可能的标题
        text = chunk_data.get('text', '')
        if text:
            lines = text.split('\n')
            if lines:
                first_line = lines[0].strip()
                # 如果第一行较短且不以句号结尾，可能是标题
                if len(first_line) < 100 and not first_line.endswith('.'):
                    return first_line
        
        return ''  # 如果都没有找到，返回空字符串
    
    def _try_fix_truncated_json(self, response: str) -> str:
        """尝试修复截断的JSON响应"""
        import re
        import json
        
        # 清理响应
        cleaned = clean_control_characters(response.strip())
        
        # 移除markdown标记
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        
        # 查找JSON开始位置
        start_idx = cleaned.find('{')
        if start_idx == -1:
            return ""
        
        # 提取从开始到最后的内容
        json_part = cleaned[start_idx:]
        
        # 如果JSON看起来被截断了（以...结尾或没有闭合括号）
        if json_part.endswith('...') or json_part.count('{') > json_part.count('}'):
            # 尝试构建一个最小的有效JSON
            try:
                # 移除...结尾
                if json_part.endswith('...'):
                    json_part = json_part[:-3]
                
                # 尝试找到content字段的值
                content_match = re.search(r'"content"\s*:\s*"([^"]*)', json_part)
                if content_match:
                    content = content_match.group(1)
                    # 构建最小的有效JSON
                    minimal_json = {
                        "content": content,
                        "keywords": [],
                        "entities": [],
                        "concepts": [],
                        "importance_score": 0.5,
                        "note_type": "fact"
                    }
                    return json.dumps(minimal_json, ensure_ascii=False)
            except Exception:
                pass
        
        return ""
    
    def _create_fallback_note(self, chunk_data: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """创建备用的原子笔记（当LLM生成失败时）"""
        # 确保 chunk_data 是字典类型
        if not isinstance(chunk_data, dict):
            if isinstance(chunk_data, str):
                chunk_data = {'text': chunk_data}
            else:
                chunk_data = {'text': str(chunk_data)}

        text = chunk_data.get('text', '')
        primary_entity = chunk_data.get('primary_entity')
        entities = TextUtils.extract_entities(text)
        if not entities and primary_entity:
            entities = [primary_entity]

        sentences = TextUtils.split_by_sentence(text)
        if sentences:
            fallback_text = sentences[0]
        else:
            fallback_text = text.strip()

        max_chars = config.get('notes_llm.max_note_chars', 200)
        if len(fallback_text) > max_chars:
            fallback_text = fallback_text[:max_chars].rstrip()

        fallback_note_data = {
            'text': fallback_text or text,
            'sent_count': 1,
            'salience': 0.5,
            'local_spans': [],
            'entities': entities,
            'years': [],
            'quality_flags': ['FALLBACK']
        }

        atomic_note = self._convert_to_atomic_note_format(fallback_note_data, chunk_data)
        atomic_note['quality_flags'] = ['FALLBACK']
        return atomic_note
    
    def _get_atomic_note_system_prompt(self) -> str:
        """获取原子笔记生成的系统提示词"""
        return ATOMIC_NOTEGEN_SYSTEM_PROMPT
    
    def _clean_list(self, items: List[str]) -> List[str]:
        """清理列表，去除空值和重复项"""
        if not isinstance(items, list):
            return []
        
        cleaned = []
        for item in items:
            if isinstance(item, str) and item.strip():
                cleaned_item = item.strip()
                if cleaned_item not in cleaned:
                    cleaned.append(cleaned_item)
        
        return cleaned
    
    def _extract_relevant_paragraph_idxs(self, text: str, paragraph_idx_mapping: Dict[str, int]) -> List[int]:
        """从文本中提取相关的paragraph idx"""
        relevant_idxs: List[int] = []

        if not paragraph_idx_mapping:
            logger.debug("paragraph_idx_mapping is empty, returning empty list")
            return relevant_idxs

        # Clean the chunk text once for all comparisons
        clean_text = TextUtils.clean_text(text)
        clean_text_lower = clean_text.lower()
        
        logger.debug(f"Matching text (length: {len(clean_text)}): {clean_text[:100]}...")
        logger.debug(f"Available paragraph mappings: {len(paragraph_idx_mapping)}")

        # 对于每个段落文本，检查是否与当前chunk的文本相关
        for paragraph_text, idx in paragraph_idx_mapping.items():
            # Clean paragraph text before any comparison
            clean_paragraph_text = TextUtils.clean_text(paragraph_text)

            # 多种匹配策略
            match_found = False
            match_strategy = ""

            # 1. 双向文本包含检查
            if clean_paragraph_text in clean_text or clean_text in clean_paragraph_text:
                match_found = True
                match_strategy = "bidirectional_containment"

            # 2. 检查段落的前100个字符是否在文本中，或文本是否在段落中
            elif len(clean_paragraph_text) > 100:
                prefix = clean_paragraph_text[:100]
                if prefix in clean_text or clean_text in clean_paragraph_text:
                    match_found = True
                    match_strategy = "prefix_matching"

            # 3. 按句子分割检查（针对长段落）
            if not match_found:
                sentences = [s.strip() for s in clean_paragraph_text.split('.') if len(s.strip()) > 30]
                for sentence in sentences[:3]:  # 只检查前3个句子
                    if sentence in clean_text or clean_text in sentence:
                        match_found = True
                        match_strategy = "sentence_matching"
                        break

            # 4. 关键词匹配（提取段落中的关键词）
            if not match_found and len(clean_paragraph_text) > 50:
                # 简单的关键词提取：长度大于5的单词
                words = [w.strip('.,!?;:"()[]{}') for w in clean_paragraph_text.split() if len(w) > 5]
                if len(words) >= 3:
                    # 如果文本中包含段落的多个关键词，认为相关
                    word_matches = sum(1 for word in words[:10] if word.lower() in clean_text_lower)
                    if word_matches >= min(3, len(words) // 2):
                        match_found = True
                        match_strategy = "keyword_matching"

            if match_found:
                relevant_idxs.append(idx)
                logger.debug(f"Found match for idx {idx} using {match_strategy}")
            else:
                logger.debug(f"No match found for idx {idx} (paragraph length: {len(clean_paragraph_text)})")

        # 去重并排序
        result = sorted(set(relevant_idxs))
        logger.debug(f"Final relevant_idxs: {result}")
        return result
    
    def _generate_stable_note_id(self, note: Dict[str, Any], fallback_index: int) -> str:
        """生成基于源文档信息的稳定note_id"""
        # 检查是否启用稳定note_id生成
        if not config.get('atomic_note_generator.stable_note_id', True):
            # 使用简单的索引方式
            return f"note_{fallback_index:06d}"
        
        import hashlib
        
        # 尝试从源信息构建稳定ID
        source_info = note.get('source_info', {})
        file_path = source_info.get('file_path', '')
        chunk_index = note.get('chunk_index', fallback_index)
        
        # 如果有文件路径信息，使用文件名+chunk_index
        if file_path:
            import os
            file_name = os.path.basename(file_path)
            # 移除文件扩展名
            file_name_base = os.path.splitext(file_name)[0]
            # 清理文件名中的特殊字符
            file_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', file_name_base)
            return f"note_{file_name_clean}_{chunk_index:06d}"
        
        # 如果有paragraph_idxs信息，使用它们生成更稳定的ID
        paragraph_idxs = note.get('paragraph_idxs', [])
        if paragraph_idxs:
            # 使用paragraph索引的哈希值
            idx_str = '_'.join(map(str, sorted(paragraph_idxs)))
            hash_obj = hashlib.md5(idx_str.encode())
            hash_short = hash_obj.hexdigest()[:8]
            return f"note_para_{hash_short}_{chunk_index:06d}"
        
        # 如果有内容，使用内容哈希
        content = note.get('content', note.get('original_text', ''))
        if content and len(content.strip()) > 0:
            # 使用内容的前100字符生成哈希
            content_sample = content.strip()[:100]
            hash_obj = hashlib.md5(content_sample.encode())
            hash_short = hash_obj.hexdigest()[:8]
            return f"note_content_{hash_short}_{chunk_index:06d}"
        
        # 最后的回退方案：使用原来的索引方式
        return f"note_{fallback_index:06d}"
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def enhance_notes_with_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强原子笔记，添加关系信息"""
        logger.info("Enhancing atomic notes with relations")
        
        for i, note in enumerate(atomic_notes):
            # 计算与其他笔记的相似度
            note['related_notes'] = []
            
            for j, other_note in enumerate(atomic_notes):
                if i != j:
                    similarity = TextUtils.calculate_similarity_keywords(
                        note['content'], other_note['content']
                    )
                    
                    if similarity > 0.3:  # 相似度阈值
                        note['related_notes'].append({
                            'note_id': other_note.get('note_id', f"note_{j:06d}"),
                            'similarity': similarity,
                            'relation_type': 'content_similarity'
                        })
            
            # 实体共现关系
            note['entity_relations'] = self._find_entity_relations(note, atomic_notes)
        
        return atomic_notes
    
    def _find_entity_relations(self, note: Dict[str, Any], all_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找实体关系"""
        relations = []
        note_entities = set(note.get('entities', []))
        
        if not note_entities:
            return relations
        
        for other_note in all_notes:
            if other_note.get('note_id') == note.get('note_id'):
                continue
            
            other_entities = set(other_note.get('entities', []))
            common_entities = note_entities.intersection(other_entities)
            
            if common_entities:
                relations.append({
                    'target_note_id': other_note.get('note_id'),
                    'common_entities': list(common_entities),
                    'relation_type': 'entity_coexistence'
                })
        
        return relations
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        
        # 计算比率
        if stats['total_chunks_processed'] > 0:
            stats['sentinel_rate'] = stats['notes_zero_count'] / stats['total_chunks_processed']
            stats['rule_fallback_rate'] = stats['rule_fallback_rate'] / stats['total_chunks_processed']
        
        return stats
    
    def reset_processing_stats(self):
        """重置处理统计信息"""
        for key in self.processing_stats:
            if isinstance(self.processing_stats[key], (int, float)):
                self.processing_stats[key] = 0
    
    def validate_atomic_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证原子笔记的质量"""
        valid_notes = []
        
        for note in atomic_notes:
            # 基本验证
            content = note.get('content', '')
            content_length = len(content.strip()) if content else 0
            
            logger.debug(f"Validating note {note.get('note_id')}: content_length={content_length}, content_preview='{content[:50] if content else 'None'}...'")
            
            if not content or content_length < 10:
                logger.warning(f"Skipping note with insufficient content: {note.get('note_id')} (length: {content_length})")
                continue
            
            # 重要性评分验证
            if note.get('importance_score', 0) < 0.1:
                logger.warning(f"Note has very low importance score: {note.get('note_id')}")
            
            valid_notes.append(note)
        
        logger.info(f"Validated {len(valid_notes)} out of {len(atomic_notes)} atomic notes")
        return valid_notes
