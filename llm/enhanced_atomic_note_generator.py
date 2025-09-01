from typing import List, Dict, Any, Optional, Union
from loguru import logger
from .local_llm import LocalLLM
from utils.batch_processor import BatchProcessor
from utils.text_utils import TextUtils
from utils.json_utils import extract_json_from_response, clean_control_characters
from utils.note_validator import NoteValidator
from utils.enhanced_ner import EnhancedNER
from utils.enhanced_relation_extractor import EnhancedRelationExtractor
from utils.enhanced_noise_filter import EnhancedNoiseFilter
from utils.note_similarity import NoteSimilarityCalculator
from config import config
from .prompts import (
    ATOMIC_NOTEGEN_SYSTEM_PROMPT,
    ATOMIC_NOTEGEN_PROMPT,
)
import json
import re
from datetime import datetime

class EnhancedAtomicNoteGenerator:
    """增强的原子笔记生成器，集成了NER优化、关系抽取、去噪机制和笔记联动功能"""
    
    def __init__(self, llm: LocalLLM = None, enable_validation: bool = True, config_override: Optional[Dict[str, Any]] = None):
        if llm is None:
            raise ValueError("EnhancedAtomicNoteGenerator requires a LocalLLM instance to be passed")
        self.llm = llm
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # 配置覆盖
        self.config = config_override or {}
        
        # 核心组件
        self.enable_validation = enable_validation
        self.validator = NoteValidator() if enable_validation else None
        
        # 增强组件
        self.enhanced_ner = EnhancedNER()
        self.relation_extractor = EnhancedRelationExtractor(local_llm=llm)
        self.noise_filter = EnhancedNoiseFilter()
        self.similarity_calculator = NoteSimilarityCalculator()
        
        # 功能开关
        self.enable_enhanced_ner = self.config.get('enable_enhanced_ner', True)
        self.enable_relation_extraction = self.config.get('enable_relation_extraction', True)
        self.enable_enhanced_noise_filter = self.config.get('enable_enhanced_noise_filter', True)
        self.enable_note_similarity = self.config.get('enable_note_similarity', False)  # Disabled due to network issues
        
        logger.info(f"Enhanced Atomic Note Generator initialized with features: "
                   f"NER={self.enable_enhanced_ner}, RE={self.enable_relation_extraction}, "
                   f"Noise={self.enable_enhanced_noise_filter}, Similarity={self.enable_note_similarity}")
    
    def generate_atomic_notes(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从文本块生成增强的原子笔记"""
        logger.info(f"Generating enhanced atomic notes for {len(text_chunks)} text chunks")
        
        # 阶段1: 基础原子笔记生成
        atomic_notes = self._generate_base_atomic_notes(text_chunks)
        
        # 阶段2: 增强NER处理
        if self.enable_enhanced_ner:
            atomic_notes = self._enhance_entities(atomic_notes)
        
        # 阶段3: 关系抽取
        if self.enable_relation_extraction:
            atomic_notes = self._extract_relations(atomic_notes)
        
        # 阶段4: 增强去噪
        if self.enable_enhanced_noise_filter:
            atomic_notes = self._filter_noise(atomic_notes)
        
        # 阶段5: 笔记相似度和关联
        if self.enable_note_similarity:
            atomic_notes = self._compute_note_similarities(atomic_notes)
        
        # 阶段6: 传统验证（如果启用）
        if self.enable_validation and self.validator:
            atomic_notes = self._validate_notes(atomic_notes, text_chunks)
        
        # 最终统计
        self._log_generation_statistics(atomic_notes)
        
        logger.info(f"Enhanced atomic note generation completed: {len(atomic_notes)} notes")
        return atomic_notes
    
    def _generate_base_atomic_notes(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成基础原子笔记"""
        logger.info("Phase 1: Generating base atomic notes")
        
        # 准备提示词模板
        system_prompt = self._get_atomic_note_system_prompt()
        
        def process_batch(batch):
            results = []
            for chunk_data in batch:
                try:
                    note = self._generate_single_atomic_note(chunk_data, system_prompt)
                    results.append(note)
                except Exception as e:
                    logger.error(f"Failed to generate atomic note: {e}")
                    # 创建基本的原子笔记
                    results.append(self._create_fallback_note(chunk_data))
            return results
        
        atomic_notes = self.batch_processor.process_batches(
            text_chunks,
            process_batch,
            desc="Generating base atomic notes"
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
        
        logger.info(f"Generated {len(atomic_notes)} base atomic notes")
        return atomic_notes
    
    def _enhance_entities(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强实体提取"""
        logger.info("Phase 2: Enhancing entity extraction")
        
        enhanced_notes = []
        for note in atomic_notes:
            # 确保note是字典类型
            if not isinstance(note, dict):
                logger.warning(f"Note in _enhance_entities is not a dict, got {type(note)}: {note}")
                if isinstance(note, list) and len(note) > 0 and isinstance(note[0], dict):
                    note = note[0]
                else:
                    note = {'content': str(note), 'error': True, 'entities': []}
            
            enhanced_note = note.copy()
            
            # 使用增强NER提取实体
            content = note.get('content', '')
            if content:
                enhanced_entities = self.enhanced_ner.extract_entities(content)
                
                # 合并原有实体和增强实体
                original_entities = set(note.get('entities', []))
                enhanced_entities_set = set(enhanced_entities)
                
                # 实体归一化
                normalized_entities = self.enhanced_ner.normalize_entities(
                    list(original_entities.union(enhanced_entities_set))
                )
                
                enhanced_note['entities'] = normalized_entities
                enhanced_note['enhanced_entities'] = enhanced_entities
                enhanced_note['entity_enhancement_applied'] = True
            
            enhanced_notes.append(enhanced_note)
        
        # 统计增强效果
        original_entity_count = sum(len(note.get('entities', [])) for note in atomic_notes)
        enhanced_entity_count = sum(len(note.get('entities', [])) for note in enhanced_notes)
        
        logger.info(f"Entity enhancement completed: {original_entity_count} -> {enhanced_entity_count} entities")
        return enhanced_notes
    
    def _extract_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽取实体关系"""
        logger.info("Phase 3: Extracting entity relations")
        
        if self.relation_extractor is None:
            logger.warning("Relation extractor not available, skipping relation extraction")
            # 为每个笔记添加空的关系信息
            for note in atomic_notes:
                note['extracted_relations'] = []
                note['relation_graph_connections'] = 0
            return atomic_notes
        
        # 批量抽取关系
        enhanced_notes = self.relation_extractor.extract_relations_from_notes(atomic_notes)
        
        # 构建关系图
        relation_graph = self.relation_extractor.build_relation_graph(enhanced_notes)
        
        # 为每个笔记添加关系图信息
        for note in enhanced_notes:
            note_id = note.get('note_id', '')
            if note_id in relation_graph:
                note['relation_graph_connections'] = len(relation_graph[note_id])
            else:
                note['relation_graph_connections'] = 0
        
        # 统计关系抽取效果
        total_relations = sum(len(note.get('extracted_relations', [])) for note in enhanced_notes)
        notes_with_relations = sum(1 for note in enhanced_notes if note.get('extracted_relations'))
        
        logger.info(f"Relation extraction completed: {total_relations} relations found in {notes_with_relations} notes")
        return enhanced_notes
    
    def _filter_noise(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强去噪处理"""
        logger.info("Phase 4: Enhanced noise filtering")
        
        # 应用增强去噪机制
        filtered_notes = self.noise_filter.filter_noise_notes(atomic_notes)
        
        # 获取过滤统计
        stats = self.noise_filter.get_filtering_statistics(atomic_notes, filtered_notes)
        
        logger.info(f"Noise filtering completed: {stats['noise_reduction']} noise notes removed, "
                   f"average usefulness score: {stats['average_usefulness_score']:.3f}")
        
        return filtered_notes
    
    def _compute_note_similarities(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算笔记相似度和关联"""
        logger.info("Phase 5: Computing note similarities and relations")
        
        # 计算笔记间的相似度和关联
        enhanced_notes = self.similarity_calculator.find_related_notes(atomic_notes)
        
        # 获取相似度统计
        similarity_stats = self.similarity_calculator.compute_similarity_statistics(enhanced_notes)
        
        logger.info(f"Note similarity computation completed: {similarity_stats['total_relations']} relations, "
                   f"coverage rate: {similarity_stats['coverage_rate']:.3f}")
        
        return enhanced_notes
    
    def _generate_single_atomic_note(self, chunk_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """生成单个原子笔记（基础版本）"""
        text = chunk_data.get('text', '')
        
        prompt = ATOMIC_NOTEGEN_PROMPT.format(text=text)
        
        response = self.llm.generate(prompt, system_prompt)
        
        try:
            # 清理响应，提取JSON部分
            cleaned_response = extract_json_from_response(response)
            
            if not cleaned_response:
                logger.warning(f"No valid JSON found in response: {response[:200]}...")
                return self._create_fallback_note(chunk_data)
            
            note_data = json.loads(cleaned_response)
            
            # 确保 note_data 是字典类型
            if not isinstance(note_data, dict):
                if isinstance(note_data, list) and len(note_data) > 0 and isinstance(note_data[0], dict):
                    logger.warning(f"LLM returned list instead of dict, using first element")
                    note_data = note_data[0]
                else:
                    logger.warning(f"Expected dict but got {type(note_data)}: {note_data}")
                    return self._create_fallback_note(chunk_data)
            
            # 提取相关的paragraph idx信息
            paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
            relevant_idxs = self._extract_relevant_paragraph_idxs(text, paragraph_idx_mapping)
            
            # 提取title和raw_span信息
            title = self._extract_title_from_chunk(chunk_data)
            raw_span = text  # raw_span就是原始文本内容
            
            # 验证和清理数据
            atomic_note = {
                'original_text': text,
                'content': note_data.get('content', text),
                'summary': note_data.get('summary', note_data.get('content', text)),  # 保留summary字段用于前端显示
                'title': title,
                'raw_span': raw_span,
                'keywords': self._clean_list(note_data.get('keywords', [])),
                'entities': self._clean_list(note_data.get('entities', [])),
                'concepts': self._clean_list(note_data.get('concepts', [])),
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
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response: {response[:200]}...")
            return self._create_fallback_note(chunk_data)
    
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
        
        # 提取title和raw_span信息
        title = self._extract_title_from_chunk(chunk_data)
        raw_span = text

        return {
            'original_text': text,
            'content': text,
            'summary': text[:100] + '...' if len(text) > 100 else text,
            'title': title,
            'raw_span': raw_span,
            'keywords': [],
            'entities': entities,
            'concepts': [],
            'normalized_entities': [],
            'normalized_predicates': [],
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

        # Clean the chunk text once for matching
        clean_text = TextUtils.clean_text(text)
        clean_text_lower = clean_text.lower()

        for paragraph_text, idx in paragraph_idx_mapping.items():
            # Clean paragraph text before any comparison
            clean_paragraph_text = TextUtils.clean_text(paragraph_text)
            clean_paragraph_lower = clean_paragraph_text.lower()
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

            # 4. 关键词重叠匹配（新增）- 处理LLM重写文本的情况
            if not match_found and len(clean_paragraph_text) > 50:
                # 提取关键词（长度大于3的单词，排除常见停用词）
                stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
                
                # 从段落中提取关键词
                para_words = [w.strip('.,!?;:"()[]{}').lower() for w in clean_paragraph_lower.split() 
                             if len(w.strip('.,!?;:"()[]{}')) > 3 and w.strip('.,!?;:"()[]{}').lower() not in stop_words]
                
                # 从生成文本中提取关键词
                text_words = [w.strip('.,!?;:"()[]{}').lower() for w in clean_text_lower.split() 
                             if len(w.strip('.,!?;:"()[]{}')) > 3 and w.strip('.,!?;:"()[]{}').lower() not in stop_words]
                
                if len(para_words) >= 5 and len(text_words) >= 5:
                    # 计算关键词重叠率
                    common_words = set(para_words[:15]) & set(text_words[:15])  # 只比较前15个关键词
                    overlap_ratio = len(common_words) / min(len(para_words[:15]), len(text_words[:15]))
                    
                    # 如果重叠率超过50%，认为匹配
                    if overlap_ratio >= 0.5:
                        match_found = True

            # 5. 实体名称匹配（新增）- 专门处理包含专有名词的情况
            if not match_found:
                # 提取可能的实体名称（大写开头的词组）
                import re
                para_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_paragraph_text)
                text_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_text)
                
                if para_entities and text_entities:
                    # 检查是否有共同的实体名称
                    common_entities = set(para_entities) & set(text_entities)
                    if len(common_entities) >= 2:  # 至少有2个共同实体
                        match_found = True

            if match_found:
                relevant_idxs.append(idx)

        # 去重并排序
        return sorted(set(relevant_idxs))
    
    def _generate_stable_note_id(self, note: Dict[str, Any], fallback_index: int) -> str:
        """生成基于源文档信息的稳定note_id"""
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
        return datetime.now().isoformat()
    
    def _validate_notes(self, atomic_notes: List[Dict[str, Any]], text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行原子笔记验证"""
        logger.info("Phase 6: Traditional validation")
        
        # 1. 实体唯一性校验
        validated_notes = self.validator.validate_entity_uniqueness(atomic_notes)
        
        # 2. 源文档实体回溯约束验证
        source_paragraphs = self._build_source_paragraphs_mapping(text_chunks)
        validated_notes = self.validator.validate_source_entity_traceability(validated_notes, source_paragraphs)
        
        # 3. 生成验证报告
        validation_report = self.validator.generate_validation_report(validated_notes)
        logger.info(f"Traditional validation completed. Report: {validation_report['summary']}")
        
        return validated_notes
    
    def _build_source_paragraphs_mapping(self, text_chunks: List[Dict[str, Any]]) -> Dict[int, str]:
        """构建源段落映射"""
        source_paragraphs = {}
        
        for chunk_data in text_chunks:
            paragraph_idx_mapping = chunk_data.get('paragraph_idx_mapping', {})
            for paragraph_text, idx in paragraph_idx_mapping.items():
                source_paragraphs[idx] = paragraph_text
        
        return source_paragraphs
    
    def _log_generation_statistics(self, atomic_notes: List[Dict[str, Any]]) -> None:
        """记录生成统计信息"""
        total_notes = len(atomic_notes)
        noise_notes = sum(1 for note in atomic_notes if note.get('is_noise', False))
        quality_notes = total_notes - noise_notes
        
        # 实体统计
        total_entities = sum(len(note.get('entities', [])) for note in atomic_notes)
        avg_entities_per_note = total_entities / max(total_notes, 1)
        
        # 关系统计
        total_relations = sum(len(note.get('extracted_relations', [])) for note in atomic_notes)
        notes_with_relations = sum(1 for note in atomic_notes if note.get('extracted_relations'))
        
        # 相似度统计
        total_similarities = sum(len(note.get('related_notes', [])) for note in atomic_notes)
        notes_with_similarities = sum(1 for note in atomic_notes if note.get('related_notes'))
        
        # 有用性评分统计
        usefulness_scores = [note.get('usefulness_score', 0) for note in atomic_notes if 'usefulness_score' in note]
        avg_usefulness = sum(usefulness_scores) / max(len(usefulness_scores), 1)
        
        logger.info(f"""Enhanced Atomic Note Generation Statistics:
        Total Notes: {total_notes}
        Quality Notes: {quality_notes} ({quality_notes/max(total_notes,1)*100:.1f}%)
        Noise Notes: {noise_notes} ({noise_notes/max(total_notes,1)*100:.1f}%)
        
        Entity Statistics:
        Total Entities: {total_entities}
        Avg Entities per Note: {avg_entities_per_note:.2f}
        
        Relation Statistics:
        Total Relations: {total_relations}
        Notes with Relations: {notes_with_relations} ({notes_with_relations/max(total_notes,1)*100:.1f}%)
        
        Similarity Statistics:
        Total Similarities: {total_similarities}
        Notes with Similarities: {notes_with_similarities} ({notes_with_similarities/max(total_notes,1)*100:.1f}%)
        
        Quality Statistics:
        Average Usefulness Score: {avg_usefulness:.3f}
        """)
    
    def get_enhancement_report(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取增强处理报告"""
        report = {
            'total_notes': len(atomic_notes),
            'enhancement_features': {
                'enhanced_ner': self.enable_enhanced_ner,
                'relation_extraction': self.enable_relation_extraction,
                'enhanced_noise_filter': self.enable_enhanced_noise_filter,
                'note_similarity': self.enable_note_similarity
            }
        }
        
        # NER增强统计
        if self.enable_enhanced_ner:
            enhanced_entity_notes = sum(1 for note in atomic_notes if note.get('entity_enhancement_applied'))
            report['ner_enhancement'] = {
                'notes_enhanced': enhanced_entity_notes,
                'enhancement_rate': enhanced_entity_notes / max(len(atomic_notes), 1)
            }
        
        # 关系抽取统计
        if self.enable_relation_extraction:
            relation_stats = self.relation_extractor.get_extraction_statistics(atomic_notes)
            report['relation_extraction'] = relation_stats
        
        # 去噪统计
        if self.enable_enhanced_noise_filter:
            noise_stats = self.noise_filter.get_filtering_statistics(atomic_notes, atomic_notes)
            report['noise_filtering'] = noise_stats
        
        # 相似度统计
        if self.enable_note_similarity:
            similarity_stats = self.similarity_calculator.compute_similarity_statistics(atomic_notes)
            report['similarity_computation'] = similarity_stats
        
        return report
    
    def export_enhanced_results(self, atomic_notes: List[Dict[str, Any]], output_dir: str) -> None:
        """导出增强处理结果"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出增强的原子笔记
        with open(os.path.join(output_dir, 'enhanced_atomic_notes.json'), 'w', encoding='utf-8') as f:
            json.dump(atomic_notes, f, ensure_ascii=False, indent=2)
        
        # 导出增强报告
        report = self.get_enhancement_report(atomic_notes)
        with open(os.path.join(output_dir, 'enhancement_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 导出关系图（如果启用了关系抽取）
        if self.enable_relation_extraction:
            relation_graph = self.relation_extractor.build_relation_graph(atomic_notes)
            with open(os.path.join(output_dir, 'relation_graph.json'), 'w', encoding='utf-8') as f:
                json.dump(relation_graph, f, ensure_ascii=False, indent=2)
        
        # 导出相似度图（如果启用了相似度计算）
        if self.enable_note_similarity:
            similarity_graph_path = os.path.join(output_dir, 'similarity_graph.json')
            self.similarity_calculator.export_similarity_graph(atomic_notes, similarity_graph_path)
        
        logger.info(f"Enhanced results exported to {output_dir}")