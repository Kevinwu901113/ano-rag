from typing import List, Dict, Any, Union, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
from .local_llm import LocalLLM
from utils.batch_processor import BatchProcessor
from utils.text_utils import TextUtils
from utils.json_utils import extract_json_from_response, clean_control_characters
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
        
        # 如果启用并发且支持并发处理，使用并发处理
        if self.concurrent_enabled and self.concurrent_support:
            client = getattr(self.llm, 'lmstudio_client', None) or getattr(self.llm, 'client', None)
            instance_count = len(getattr(client, 'model_instances', getattr(client, 'instances', []))) if client else 1
            logger.info(f"Using concurrent processing with {instance_count} instances")
            return self._generate_atomic_notes_concurrent(text_chunks, progress_tracker)
        else:
            logger.info("Using sequential processing")
            return self._generate_atomic_notes_sequential(text_chunks, progress_tracker)
    
    def _generate_atomic_notes_sequential(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """顺序生成原子笔记（原有逻辑）"""
        # 准备提示词模板
        system_prompt = self._get_atomic_note_system_prompt()
        
        def process_batch(batch):
            # 检查batch是否是单个item（当batch_processor回退到单个处理时）
            if not isinstance(batch, list):
                # 单个item处理，直接返回单个结果而不是列表
                try:
                    note = self._generate_single_atomic_note(batch, system_prompt)
                    return note
                except Exception as e:
                    logger.error(f"Failed to generate atomic note: {e}")
                    return self._create_fallback_note(batch)
            
            # 批处理多个items，支持每chunk N笔记的容器结构
            results = []
            for chunk_data in batch:
                try:
                    notes = self._generate_single_atomic_note(chunk_data, system_prompt)
                    
                    # 处理返回的笔记（可能是单个笔记或笔记列表）
                    if isinstance(notes, list):
                        # 新结构：多个笔记
                        for note in notes:
                            if isinstance(note, dict):
                                results.append(note)
                    else:
                        # 旧结构：单个笔记
                        results.append(notes)
                        
                        # 检查是否有额外的原子笔记需要处理（旧结构兼容）
                        if '_additional_notes' in chunk_data:
                            additional_notes = chunk_data.pop('_additional_notes')
                            for additional_note_data in additional_notes:
                                try:
                                    # 为每个额外的笔记创建完整的原子笔记
                                    additional_note = self._create_atomic_note_from_data(
                                        additional_note_data, chunk_data
                                    )
                                    results.append(additional_note)
                                except Exception as e:
                                    logger.error(f"Failed to process additional note: {e}")
                                
                except Exception as e:
                    logger.error(f"Failed to generate atomic note: {e}")
                    # 创建基本的原子笔记
                    results.append(self._create_fallback_note(chunk_data))
            
            # 应用去重策略
            results = self._apply_deduplication(results)
            
            # 应用配额与阈值筛选
            results = self._apply_quota_and_threshold(results)
            
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
    
    def _apply_deduplication(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用去重策略，基于字符跨度+文本或文本hash进行去重"""
        if not notes:
            return notes
        
        import hashlib
        seen_hashes = set()
        deduplicated_notes = []
        
        for note in notes:
            # 生成去重hash
            dedup_hash = self._generate_dedup_hash(note)
            
            if dedup_hash not in seen_hashes:
                seen_hashes.add(dedup_hash)
                deduplicated_notes.append(note)
            else:
                logger.debug(f"Duplicate note filtered: {note.get('content', '')[:50]}...")
        
        logger.info(f"Deduplication: {len(notes)} -> {len(deduplicated_notes)} notes")
        return deduplicated_notes
    
    def _generate_dedup_hash(self, note: Dict[str, Any]) -> str:
        """生成去重hash，根据配置选择去重策略"""
        import hashlib
        
        # 获取去重策略配置
        dedup_by = config.get('atomic_note_generator.dedup_by', 'span_or_hash')
        
        char_span = note.get('char_span')
        text = note.get('content', note.get('original_text', ''))
        
        if dedup_by == 'span_only':
            # 仅使用字符跨度去重
            if char_span and isinstance(char_span, (list, tuple)) and len(char_span) == 2:
                span_text = f"{char_span[0]}:{char_span[1]}"
                return hashlib.md5(span_text.encode('utf-8')).hexdigest()
            else:
                # 如果没有span信息，回退到文本hash
                return self._generate_content_hash(text)
        
        elif dedup_by == 'content_hash':
            # 仅使用内容hash去重
            return self._generate_content_hash(text)
        
        else:  # dedup_by == 'span_or_hash' (默认)
            # 优先使用字符跨度+文本，回退到文本hash
            if char_span and isinstance(char_span, (list, tuple)) and len(char_span) == 2:
                # 使用字符跨度+文本内容生成hash
                span_text = f"{char_span[0]}:{char_span[1]}:{text}"
                return hashlib.md5(span_text.encode('utf-8')).hexdigest()
            else:
                # 回退到文本内容hash
                return self._generate_content_hash(text)
    
    def _generate_content_hash(self, text: str) -> str:
        """生成文本内容的hash"""
        import hashlib
        
        if text:
            # 标准化文本（去除多余空白、统一大小写）
            normalized_text = ' '.join(text.strip().lower().split())
            return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
        
        # 如果没有文本内容，返回空字符串的hash
        return hashlib.md5(''.encode('utf-8')).hexdigest()
    
    def _apply_quota_and_threshold(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用配额与阈值筛选"""
        if not notes:
            return notes
        
        # 获取配置参数
        max_facts_per_sentence = config.get('atomic_note_generator.max_facts_per_sentence', 5)
        min_fact_length = config.get('atomic_note_generator.min_fact_len', 10)
        min_importance_score = config.get('atomic_note_generator.min_importance', 0.1)
        
        filtered_notes = []
        sentence_fact_counts = {}
        
        # 按重要性分数排序，优先保留高分事实
        sorted_notes = sorted(notes, key=lambda x: x.get('importance_score', 0.5), reverse=True)
        
        for note in sorted_notes:
            # 检查最小长度
            content = note.get('content', '')
            if len(content.strip()) < min_fact_length:
                logger.debug(f"Note filtered by min length: {content[:30]}...")
                continue
            
            # 检查最小重要性分
            importance_score = note.get('importance_score', 0.5)
            if importance_score < min_importance_score:
                logger.debug(f"Note filtered by min importance: {importance_score}")
                continue
            
            # 检查每句最大事实数（基于句序号）
            sentence_index = note.get('sentence_index', 0)
            current_count = sentence_fact_counts.get(sentence_index, 0)
            
            if current_count >= max_facts_per_sentence:
                logger.debug(f"Note filtered by sentence quota: sentence {sentence_index}")
                continue
            
            # 通过所有筛选条件
            filtered_notes.append(note)
            sentence_fact_counts[sentence_index] = current_count + 1
        
        logger.info(f"Quota/threshold filtering: {len(notes)} -> {len(filtered_notes)} notes")
        return filtered_notes
    
    def _generate_atomic_notes_concurrent(self, text_chunks: List[Dict[str, Any]], progress_tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """并发生成原子笔记，利用多个LM Studio实例"""
        system_prompt = self._get_atomic_note_system_prompt()
        atomic_notes = [None] * len(text_chunks)  # 预分配结果列表
        
        # 计算实际的并发工作线程数
        client = getattr(self.llm, 'lmstudio_client', None) or getattr(self.llm, 'client', None)
        instance_count = len(getattr(client, 'model_instances', getattr(client, 'instances', []))) if client else 1
        max_workers = min(self.max_concurrent_workers, instance_count, len(text_chunks))
        
        logger.info(f"Starting concurrent processing with {max_workers} workers for {len(text_chunks)} chunks")
        
        def process_chunk_with_index(chunk_index_pair):
            """处理单个文本块并返回索引和结果（支持多笔记返回）"""
            chunk_data, index = chunk_index_pair
            try:
                notes = self._generate_single_atomic_note(chunk_data, system_prompt)
                return index, notes, None
            except Exception as e:
                logger.error(f"Failed to generate atomic note for chunk {index}: {e}")
                fallback_note = self._create_fallback_note(chunk_data)
                return index, fallback_note, e
        
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
            
            all_notes = []  # 存储所有生成的笔记
            chunk_to_notes_mapping = {}  # 映射chunk索引到笔记列表
            
            for future in as_completed(future_to_index):
                try:
                    index, notes, error = future.result()
                    
                    # 处理返回的笔记（可能是单个笔记或笔记列表）
                    if isinstance(notes, list):
                        # 新结构：多个笔记
                        chunk_notes = []
                        for note in notes:
                            if isinstance(note, dict):
                                chunk_notes.append(note)
                                all_notes.append(note)
                        chunk_to_notes_mapping[index] = chunk_notes
                    else:
                        # 旧结构：单个笔记
                        chunk_to_notes_mapping[index] = [notes]
                        all_notes.append(notes)
                    
                    if error:
                        error_count += 1
                    
                    completed_count += 1
                    
                    # 更新进度跟踪器
                    if progress_tracker:
                        progress_tracker.update(1)
                    
                    if completed_count % 10 == 0:  # 每10个任务记录一次进度
                        logger.info(f"Completed {completed_count}/{len(text_chunks)} chunks, generated {len(all_notes)} notes")
                        
                except Exception as e:
                    original_index = future_to_index[future]
                    logger.error(f"Future execution failed for chunk {original_index}: {e}")
                    # 创建fallback note
                    if original_index < len(text_chunks):
                        fallback_note = self._create_fallback_note(text_chunks[original_index])
                        chunk_to_notes_mapping[original_index] = [fallback_note]
                        all_notes.append(fallback_note)
                    error_count += 1
                    
                    # 更新进度跟踪器（即使出错也要更新）
                    if progress_tracker:
                        progress_tracker.update(1)
        
        logger.info(f"Concurrent processing completed: {len(text_chunks)} chunks processed, generated {len(all_notes)} notes, {error_count} errors")
        
        # 应用去重策略
        if all_notes:
            all_notes = self._apply_deduplication(all_notes)
        
        # 应用配额与阈值筛选
        if all_notes:
            all_notes = self._apply_quota_and_threshold(all_notes)
        
        # 后处理：添加ID和元数据
        for i, note in enumerate(all_notes):
            # 确保note是字典类型
            if not isinstance(note, dict):
                logger.warning(f"Note at index {i} is not a dict, got {type(note)}: {note}")
                note = {'content': str(note), 'error': True}
                all_notes[i] = note
            
            note['note_id'] = self._generate_stable_note_id(note, i)
            note['created_at'] = self._get_timestamp()
        
        # 摘要校验：仅在启用时进行
        if config.get('summary_auditor.enabled', False):
            try:
                from utils.summary_auditor import SummaryAuditor
                logger.info("Starting summary audit for generated atomic notes")
                auditor = SummaryAuditor(llm=self.llm)
                all_notes = auditor.batch_audit_summaries(all_notes)
                logger.info("Summary audit completed")
            except ImportError as e:
                logger.warning(f"Failed to import SummaryAuditor: {e}")
            except Exception as e:
                logger.error(f"Summary audit failed: {e}")
        
        logger.info(f"Generated {len(all_notes)} atomic notes")
        return all_notes
    
    def _generate_single_atomic_note(self, chunk_data: Union[Dict[str, Any], Any], system_prompt: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """生成单个或多个原子笔记，支持新旧两种返回结构"""
        # 确保 chunk_data 是字典类型
        if not isinstance(chunk_data, dict):
            logger.warning(f"chunk_data is not a dict, got {type(chunk_data)}: {chunk_data}")
            # 如果是字符串，创建基本的字典结构
            if isinstance(chunk_data, str):
                chunk_data = {'text': chunk_data}
            else:
                logger.error(f"Unsupported chunk_data type: {type(chunk_data)}")
                return self._create_fallback_note({'text': str(chunk_data)})
        
        text = chunk_data.get('text', '')
        
        prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
        
        # 根据模式选择生成器
        if self.is_hybrid_mode and self.hybrid_dispatcher:
            response = self.hybrid_dispatcher.process_single(prompt, system_prompt)
        else:
            response = self.llm.generate(prompt, system_prompt)
        
        # 解析响应，兼容新旧两种结构
        return self._parse_response_with_compatibility(response, chunk_data)
    
    def _parse_response_with_compatibility(self, response: str, chunk_data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """解析响应，兼容新旧两种结构"""
        try:
            import json
            import re
            
            # 清理响应，提取JSON部分
            cleaned_response = extract_json_from_response(response)
            
            if not cleaned_response:
                logger.warning(f"No valid JSON found in response: {response[:200]}...")
                return self._create_fallback_note(chunk_data)
            
            note_data = json.loads(cleaned_response)
            
            # 检查是否为新结构（按句分组）
            if self._is_new_structure(note_data):
                return self._parse_new_structure(note_data, chunk_data)
            else:
                return self._parse_old_structure(note_data, chunk_data)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response: {response[:200]}...")
            return self._create_fallback_note(chunk_data)
    
    def _is_new_structure(self, note_data: Any) -> bool:
        """检查是否为新结构（按句多事实数组）"""
        # 检查是否为数组，且数组元素包含facts字段
        if isinstance(note_data, list) and len(note_data) > 0:
            # 检查第一个元素是否包含facts字段
            first_item = note_data[0]
            if isinstance(first_item, dict) and 'facts' in first_item:
                return True
        return False
    
    def _parse_new_structure(self, note_data: List[Dict[str, Any]], chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析新结构（按句多事实数组）"""
        if not isinstance(note_data, list):
            logger.warning("note_data is not a list, falling back to old structure")
            return [self._parse_old_structure(note_data, chunk_data)]
        
        atomic_notes = []
        total_facts = 0
        
        # 获取配置参数
        max_facts_per_sentence = config.get('atomic_note_generator.max_facts_per_sentence', 5)
        
        for sentence_item in note_data:
            if not isinstance(sentence_item, dict):
                continue
                
            sent_id = sentence_item.get('sent_id', 0)
            facts = sentence_item.get('facts', [])
            
            if not isinstance(facts, list):
                continue
                
            # 应用每句事实数量限制
            if len(facts) > max_facts_per_sentence:
                logger.debug(f"Sentence {sent_id}: limiting facts from {len(facts)} to {max_facts_per_sentence}")
                facts = facts[:max_facts_per_sentence]
            
            # 为每个事实创建一个原子笔记
            for fact_idx, fact in enumerate(facts):
                if not isinstance(fact, dict):
                    continue
                    
                # 创建原子笔记，保留所有元数据
                atomic_note = self._create_atomic_note_from_fact(fact, sentence_item, chunk_data, fact_idx)
                atomic_notes.append(atomic_note)
                total_facts += 1
        
        sentences_with_facts = len([item for item in note_data if isinstance(item.get('facts'), list) and len(item.get('facts', [])) > 0])
        avg_facts_per_sentence = total_facts / max(sentences_with_facts, 1)
        
        logger.info(f"Parsed {total_facts} atomic notes from {len(note_data)} sentences (avg: {avg_facts_per_sentence:.2f} facts/sentence)")
        return atomic_notes
    
    def _parse_old_structure(self, note_data: Any, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析旧结构（单对象）"""
        # 处理 note_data 的不同类型
        if isinstance(note_data, list):
            # 如果是列表，处理所有有效的字典项
            valid_notes = [item for item in note_data if isinstance(item, dict)]
            if valid_notes:
                logger.info(f"Processing {len(valid_notes)} atomic notes from list")
                # 处理第一个笔记，其余的将在后续处理中返回
                note_data = valid_notes[0]
                # 如果有多个笔记，我们需要特殊处理
                if len(valid_notes) > 1:
                    # 存储额外的笔记以便后续处理
                    chunk_data['_additional_notes'] = valid_notes[1:]
            else:
                logger.warning("No valid dict found in list, creating fallback note")
                return self._create_fallback_note(chunk_data)
        elif not isinstance(note_data, dict):
            logger.warning(f"note_data is not a dict or list, got {type(note_data)}: {note_data}")
            return self._create_fallback_note(chunk_data)
        
        # 使用统一的方法创建原子笔记
        return self._create_atomic_note_from_data(note_data, chunk_data)
    
    def _create_atomic_note_from_fact(self, fact: Dict[str, Any], sentence_item: Dict[str, Any], chunk_data: Dict[str, Any], fact_idx: int) -> Dict[str, Any]:
        """从单个事实创建原子笔记，保留所有元数据"""
        text = chunk_data.get('text', '')
        
        # 提取相关的paragraph idx信息
        paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
        
        # 从事实中提取信息
        fact_text = fact.get('text', '')
        entities = fact.get('entities', [])
        predicate = fact.get('pred', '')
        time_info = fact.get('time', '')
        importance_score = fact.get('score', 0.5)
        fact_type = fact.get('type', 'fact')
        span = fact.get('span', [])
        
        # 从句子项中获取sent_id
        sent_id = sentence_item.get('sent_id', 0)
        
        # 生成原子笔记
        atomic_note = {
            'text': fact_text,
            'entities': entities if isinstance(entities, list) else [],
            'relations': [],  # 将在后续处理中填充
            'predicate': predicate,
            'time': time_info,
            'importance_score': float(importance_score) if importance_score else 0.5,
            'fact_type': fact_type,
            'sent_id': sent_id,
            'fact_idx': fact_idx,
            'span': span if isinstance(span, list) and len(span) == 2 else [],
            
            # 保留原有字段
            'chunk_id': chunk_data.get('chunk_id', ''),
            'doc_id': chunk_data.get('doc_id', ''),
            'paragraph_idx': chunk_data.get('paragraph_idx', 0),
            'paragraph_idx_mapping': paragraph_idx_mapping,
            'chunk_text': text,
            'source': chunk_data.get('source', ''),
            'metadata': chunk_data.get('metadata', {})
        }
        
        # 生成关系信息（如果有谓词和实体）
        if predicate and len(entities) >= 2:
            atomic_note['relations'] = [{
                'source': entities[0],
                'target': entities[1],
                'relation_type': predicate
            }]
        
        # 生成span evidence
        if entities:
            atomic_note['span_evidence'] = self._generate_raw_span_evidence(entities, atomic_note['relations'], text)
        else:
            atomic_note['span_evidence'] = fact_text[:100] if fact_text else text[:100]
        
        return atomic_note
    
    def _create_atomic_note_from_data(self, note_data: Dict[str, Any], chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """从note_data和chunk_data创建完整的原子笔记，支持新结构字段"""
        text = chunk_data.get('text', '')
        
        # 提取相关的paragraph idx信息
        paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
        relevant_idxs = self._extract_relevant_paragraph_idxs(text, paragraph_idx_mapping)
        
        # 提取title和raw_span信息
        title = self._extract_title_from_chunk(chunk_data)
        raw_span = text  # raw_span就是原始文本内容
        
        # 提取实体和关系信息
        entities = self._clean_list(note_data.get('entities', []))
        relations = note_data.get('relations', [])
        
        # 生成raw_span_evidence
        raw_span_evidence = self._generate_raw_span_evidence(entities, relations, text)
        
        # 验证和清理数据
        atomic_note = {
            'original_text': text,
            'content': note_data.get('content', text),
            'summary': note_data.get('summary', note_data.get('content', text)),  # 保留summary字段用于前端显示
            'title': title,
            'raw_span': raw_span,
            'raw_span_evidence': raw_span_evidence,  # 新增字段
            'keywords': self._clean_list(note_data.get('keywords', [])),
            'entities': entities,
            'concepts': self._clean_list(note_data.get('concepts', [])),
            'relations': relations if isinstance(relations, list) else [],  # 确保relations字段存在
            'normalized_entities': self._clean_list(note_data.get('normalized_entities', [])),
            'normalized_predicates': self._clean_list(note_data.get('normalized_predicates', [])),
            'importance_score': float(note_data.get('importance_score', 0.5)),
            'note_type': note_data.get('note_type', 'fact'),
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text),
            'paragraph_idxs': relevant_idxs,
            # 新结构支持的字段
            'sentence_index': note_data.get('sentence_index', 0),  # 句序号
            'timestamp': note_data.get('timestamp', ''),  # 时间信息
            'predicate': note_data.get('predicate', ''),  # 谓词
            'char_span': note_data.get('char_span'),  # 字符跨度
            'group_index': note_data.get('group_index', 0)  # 分组索引
        }
        
        # 提取额外的实体（如果LLM没有提取到）
        if not atomic_note['entities']:
            atomic_note['entities'] = TextUtils.extract_entities(text)
        
        # 确保包含主要实体
        primary_entity = chunk_data.get('primary_entity')
        if primary_entity and primary_entity not in atomic_note['entities']:
            atomic_note['entities'].insert(0, primary_entity)
        
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
        """创建统一的回退笔记（当LLM生成失败或返回为空时）"""
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
        
        # 生成宽泛的内容摘要
        fallback_content = self._generate_fallback_content(text)
        
        # 提取相关的paragraph idx信息
        paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
        relevant_idxs = self._extract_relevant_paragraph_idxs(text, paragraph_idx_mapping)
        
        # 提取title信息
        title = self._extract_title_from_chunk(chunk_data)

        return {
            'original_text': text,
            'content': fallback_content,
            'summary': fallback_content,
            'title': title,
            'raw_span': text,
            'raw_span_evidence': f"Content from {title or 'document'}" if title else text[:100],
            'keywords': self._extract_basic_keywords(text),
            'entities': entities,
            'concepts': [],
            'relations': [],
            'normalized_entities': [],
            'normalized_predicates': [],
            'importance_score': config.get('atomic_note_generator.fallback_importance_score', 0.2),  # 低权重
            'note_type': 'fallback',
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text),
            'paragraph_idxs': relevant_idxs,
            # 新结构字段的默认值
            'sentence_index': 0,
            'timestamp': '',
            'predicate': '',
            'char_span': None,
            'group_index': 0,
            'is_fallback': True  # 标记为回退笔记
        }
    
    def _generate_fallback_content(self, text: str) -> str:
        """生成回退内容的宽泛摘要"""
        if not text or len(text.strip()) == 0:
            return "Empty content"
        
        # 如果文本很短，直接返回
        if len(text) <= 100:
            return text.strip()
        
        # 提取前几句话作为摘要
        sentences = text.split('.')[:2]  # 取前两句
        summary = '. '.join(s.strip() for s in sentences if s.strip())
        
        if summary:
            return summary + ('.' if not summary.endswith('.') else '')
        
        # 如果没有句子分隔符，返回前100个字符
        return text[:100].strip() + '...'
    
    def _extract_basic_keywords(self, text: str) -> List[str]:
        """提取基本关键词"""
        if not text:
            return []
        
        import re
        # 简单的关键词提取：长度大于3的单词，去除常见停用词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # 去重并限制数量
        unique_keywords = list(dict.fromkeys(keywords))[:5]
        return unique_keywords
    
    def _get_atomic_note_system_prompt(self) -> str:
        """获取原子笔记生成的系统提示词，根据include_span配置动态调整"""
        # 检查是否启用字符跨度功能
        include_span = config.get('atomic_note_generator.include_span', False)
        
        if include_span:
            # 如果启用字符跨度，使用包含跨度信息的提示词
            span_instruction = "\n\n请在每个原子笔记中包含char_span字段，格式为[start_pos, end_pos]，表示该笔记内容在原文中的字符位置范围。"
            return ATOMIC_NOTEGEN_SYSTEM_PROMPT + span_instruction
        else:
            # 使用标准提示词
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
            return relevant_idxs

        # Clean the chunk text once for all comparisons
        clean_text = TextUtils.clean_text(text)
        clean_text_lower = clean_text.lower()

        # 对于每个段落文本，检查是否与当前chunk的文本相关
        for paragraph_text, idx in paragraph_idx_mapping.items():
            # Clean paragraph text before any comparison
            clean_paragraph_text = TextUtils.clean_text(paragraph_text)

            # 多种匹配策略
            match_found = False

            # 1. 双向文本包含检查
            if clean_paragraph_text in clean_text or clean_text in clean_paragraph_text:
                match_found = True

            # 2. 检查段落的前100个字符是否在文本中，或文本是否在段落中
            elif len(clean_paragraph_text) > 100:
                prefix = clean_paragraph_text[:100]
                if prefix in clean_text or clean_text in clean_paragraph_text:
                    match_found = True

            # 3. 按句子分割检查（针对长段落）
            if not match_found:
                sentences = [s.strip() for s in clean_paragraph_text.split('.') if len(s.strip()) > 30]
                for sentence in sentences[:3]:  # 只检查前3个句子
                    if sentence in clean_text or clean_text in sentence:
                        match_found = True
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

            if match_found:
                relevant_idxs.append(idx)

        # 去重并排序
        return sorted(set(relevant_idxs))
    
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
    
    def validate_atomic_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证原子笔记的质量"""
        valid_notes = []
        
        for note in atomic_notes:
            # 基本验证
            if not note.get('content') or len(note['content'].strip()) < 10:
                logger.warning(f"Skipping note with insufficient content: {note.get('note_id')}")
                continue
            
            # 重要性评分验证
            if note.get('importance_score', 0) < 0.1:
                logger.warning(f"Note has very low importance score: {note.get('note_id')}")
            
            valid_notes.append(note)
        
        logger.info(f"Validated {len(valid_notes)} out of {len(atomic_notes)} atomic notes")
        return valid_notes
    
    def _get_atomic_note_system_prompt(self) -> str:
        """获取原子笔记生成的系统提示词，根据include_span配置动态调整"""
        # 检查是否启用字符跨度功能
        include_span = config.get('atomic_note_generator.include_span', False)
        
        if include_span:
            # 如果启用字符跨度，使用包含跨度信息的提示词
            span_instruction = "\n\n请在每个原子笔记中包含char_span字段，格式为[start_pos, end_pos]，表示该笔记内容在原文中的字符位置范围。"
            return ATOMIC_NOTEGEN_SYSTEM_PROMPT + span_instruction
        else:
            # 使用标准提示词
            return ATOMIC_NOTEGEN_SYSTEM_PROMPT
