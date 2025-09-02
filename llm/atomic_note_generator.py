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
            
            # 批处理多个items
            results = []
            for chunk_data in batch:
                try:
                    note = self._generate_single_atomic_note(chunk_data, system_prompt)
                    results.append(note)
                    
                    # 检查是否有额外的原子笔记需要处理
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
        atomic_notes = [None] * len(text_chunks)  # 预分配结果列表
        
        # 计算实际的并发工作线程数
        client = getattr(self.llm, 'lmstudio_client', None) or getattr(self.llm, 'client', None)
        instance_count = len(getattr(client, 'model_instances', getattr(client, 'instances', []))) if client else 1
        max_workers = min(self.max_concurrent_workers, instance_count, len(text_chunks))
        
        logger.info(f"Starting concurrent processing with {max_workers} workers for {len(text_chunks)} chunks")
        
        def process_chunk_with_index(chunk_index_pair):
            """处理单个文本块并返回索引和结果"""
            chunk_data, index = chunk_index_pair
            try:
                note = self._generate_single_atomic_note(chunk_data, system_prompt)
                return index, note, None
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
            
            for future in as_completed(future_to_index):
                try:
                    index, note, error = future.result()
                    atomic_notes[index] = note
                    
                    if error:
                        error_count += 1
                    
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
                        atomic_notes[original_index] = self._create_fallback_note(text_chunks[original_index])
                    error_count += 1
                    
                    # 更新进度跟踪器（即使出错也要更新）
                    if progress_tracker:
                        progress_tracker.update(1)
        
        logger.info(f"Concurrent processing completed: {len(text_chunks)} notes generated, {error_count} errors")
        
        # 后处理：添加ID和元数据
        for i, note in enumerate(atomic_notes):
            if note is None:
                # 如果某个位置没有结果，创建fallback note
                note = self._create_fallback_note(text_chunks[i] if i < len(text_chunks) else {'text': ''})
                atomic_notes[i] = note
            
            # 确保note是字典类型
            if not isinstance(note, dict):
                logger.warning(f"Note at index {i} is not a dict, got {type(note)}: {note}")
                note = {'content': str(note), 'error': True}
                atomic_notes[i] = note
            
            note['note_id'] = f"note_{i:06d}"
            note['created_at'] = self._get_timestamp()
        
        # 摘要校验：仅在启用时进行
        if config.get('summary_auditor.enabled', False):
            try:
                from utils.summary_auditor import SummaryAuditor
                logger.info("Starting summary audit for generated atomic notes")
                auditor = SummaryAuditor(llm=self.llm)
                atomic_notes = auditor.audit_atomic_notes(atomic_notes)
            except Exception as e:
                logger.error(f"Summary audit failed: {e}")
        
        return atomic_notes
    
    def _generate_single_atomic_note(self, chunk_data: Union[Dict[str, Any], Any], system_prompt: str) -> Dict[str, Any]:
        """生成单个原子笔记"""
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
        
        response = self.llm.generate(prompt, system_prompt)
        
        try:
            import json
            import re
            
            # 清理响应，提取JSON部分
            cleaned_response = extract_json_from_response(response)
            
            if not cleaned_response:
                logger.warning(f"No valid JSON found in response: {response[:200]}...")
                return self._create_fallback_note(chunk_data)
            
            note_data = json.loads(cleaned_response)
            
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
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response: {response[:200]}...")
            return self._create_fallback_note(chunk_data)
    
    def _create_atomic_note_from_data(self, note_data: Dict[str, Any], chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """从note_data和chunk_data创建完整的原子笔记"""
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
            'paragraph_idxs': relevant_idxs
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

        return {
            'original_text': text,
            'content': text,
            'keywords': [],
            'entities': entities,
            'concepts': [],
            'importance_score': 0.5,
            'note_type': 'fact',
            'source_info': chunk_data.get('source_info', {}),
            'chunk_index': chunk_data.get('chunk_index', 0),
            'length': len(text)
        }
    
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
