from typing import List, Dict, Any, Optional, Union
from loguru import logger
from utils import FileUtils, TextUtils
from config import config
import os
from pathlib import Path

class DocumentChunker:
    """文档分块器，用于将文档分割成合适的块"""
    
    def __init__(self):
        self.chunk_size = config.get('document.chunk_size', 512)
        self.overlap = config.get('document.overlap', 50)
        self.supported_formats = config.get('document.supported_formats', ['json', 'jsonl', 'docx'])
        
        # 事件链优化配置
        self.event_chain_optimization = config.get('chunking.event_chain_optimization.enabled', True)
        self.event_keywords = {
            'succession': ['继任', '接任', '接替', '继承', '接班', '替代', '取代', 'succeed', 'replace', 'take over'],
            'acquisition': ['收购', '并购', '兼并', '购买', '买下', 'acquire', 'purchase', 'buy out', 'takeover'],
            'ownership': ['拥有', '持有', '控制', '所有', '归属', 'own', 'control', 'possess', 'belong to'],
            'bankruptcy': ['破产', '倒闭', '清算', '解散', 'bankruptcy', 'liquidation', 'dissolution'],
            'merger': ['合并', '融合', '整合', 'merge', 'consolidate', 'integrate'],
            'partnership': ['合作', '合伙', '联盟', '伙伴', 'partner', 'collaborate', 'alliance']
        }

        # Store raw paragraph text for diagnostics when building idx mappings
        self.paragraph_original_texts: Dict[int, str] = {}
        
    def chunk_document(self, file_path: str, source_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """对单个文档进行分块"""
        logger.info(f"Chunking document: {file_path}")
        
        try:
            # 读取文档内容
            content = self._read_document_content(file_path)
            
            # 特殊处理：如果是musique格式的JSONL数据，需要提取idx信息和段落信息
            paragraph_idx_mapping = None
            paragraph_info = []
            
            # 处理JSONL格式：如果content是列表，取第一个元素
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict) and 'paragraphs' in first_item:
                    paragraph_idx_mapping = self._extract_paragraph_idx_mapping(first_item)
                    paragraph_info = self._extract_paragraph_info(first_item)
                    # 使用第一个JSONL条目作为内容
                    content = first_item
            elif isinstance(content, dict) and 'paragraphs' in content:
                paragraph_idx_mapping = self._extract_paragraph_idx_mapping(content)
                paragraph_info = self._extract_paragraph_info(content)
            
            # 提取文本内容
            text_content = self._extract_text_content(content, file_path)
            
            # === 新增：当有段落信息时，按段落切块并写入 paragraph_idx ===
            if paragraph_info:
                chunks = []
                global_idx = 0  # 跨全文的 chunk 序号

                effective_source_info = source_info or {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_hash': FileUtils.get_file_hash(file_path) if hasattr(FileUtils, 'get_file_hash') else 'unknown'
                }

                for para in paragraph_info:
                    para_text = (para.get('paragraph_text') or '').strip()
                    if not para_text:
                        continue

                    sentence_chunks = self._chunk_paragraph_by_sentence(para_text, effective_source_info)
                    if not sentence_chunks:
                        continue

                    mapped_idx = para.get('idx')
                    try:
                        mapped_idx = int(str(mapped_idx).strip()) if mapped_idx is not None else None
                    except Exception:
                        mapped_idx = None

                    local_key = TextUtils.clean_text(para_text)[:100]
                    local_mapping = {local_key: mapped_idx} if mapped_idx is not None else {}

                    for j, sc in enumerate(sentence_chunks):
                        sc['chunk_index'] = global_idx
                        sc['para_local_chunk_index'] = j
                        sc['paragraph_idx'] = mapped_idx
                        sc['paragraph_info'] = [para]
                        sc['paragraph_idx_mapping'] = local_mapping
                        sc['chunk_id'] = (
                            f"{effective_source_info.get('file_name', 'unknown')}"
                            f"_p{mapped_idx if mapped_idx is not None else 'x'}_{j:03d}"
                        )
                        chunks.append(sc)
                        global_idx += 1

                logger.info(f"Document chunked by paragraph: {len(chunks)} chunks")
                return chunks

            # 分块处理
            chunks = self._chunk_text_content(text_content, file_path, source_info)

            # 为每个chunk添加paragraph_idx_mapping和paragraph_info信息
            if paragraph_idx_mapping:
                for chunk in chunks:
                    chunk['paragraph_idx_mapping'] = paragraph_idx_mapping

            if paragraph_info:
                for chunk in chunks:
                    chunk['paragraph_info'] = paragraph_info

            logger.info(f"Document chunked into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk document {file_path}: {e}")
            return []
    
    def chunk_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量处理多个文档"""
        all_chunks = []
        
        for file_path in file_paths:
            source_info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_hash': FileUtils.get_file_hash(file_path)
            }
            
            chunks = self.chunk_document(file_path, source_info)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _read_document_content(self, file_path: str) -> Union[str, Dict, List]:
        """读取文档内容"""
        try:
            return FileUtils.read_document(file_path)
        except Exception as e:
            logger.error(f"Failed to read document {file_path}: {e}")
            raise
    
    def _extract_text_content(self, content: Union[str, Dict, List], file_path: str) -> str:
        """从不同格式的内容中提取文本"""
        if isinstance(content, str):
            return content
        
        elif isinstance(content, dict):
            return self._extract_text_from_dict(content)
        
        elif isinstance(content, list):
            return self._extract_text_from_list(content)
        
        else:
            logger.warning(f"Unknown content type for {file_path}: {type(content)}")
            return str(content)
    
    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """从字典中提取文本内容"""
        text_parts = []
        
        # 处理musique数据集格式：包含paragraphs字段（可能包含或不包含question字段）
        if 'paragraphs' in data:
            # 只提取段落文本，不包含问题部分
            paragraphs = data.get('paragraphs', [])
            for para in paragraphs:
                if isinstance(para, dict) and 'paragraph_text' in para:
                    text_parts.append(para['paragraph_text'])
            
            # 注意：不添加question部分，原子笔记应该只来源于paragraphs
        else:
            # 常见的文本字段
            text_fields = ['text', 'content', 'body', 'description', 'summary', 'title']
            
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    text_parts.append(data[field])
            
            # 如果没有找到标准字段，尝试提取所有字符串值
            if not text_parts:
                for key, value in data.items():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        text_parts.append(f"{key}: {value}")
        
        return '\n'.join(text_parts)
    
    def _extract_paragraph_info(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从数据中提取段落信息，包括title和paragraph_text"""
        paragraph_info = []
        
        if 'paragraphs' in data:
            paragraphs = data.get('paragraphs', [])
            for para in paragraphs:
                if isinstance(para, dict) and 'paragraph_text' in para:
                    info = {
                        'title': para.get('title', ''),
                        'paragraph_text': para.get('paragraph_text', ''),
                        'idx': para.get('idx', -1)
                    }
                    paragraph_info.append(info)
        
        return paragraph_info
    
    def _extract_text_from_list(self, data: List[Any]) -> str:
        """从列表中提取文本内容"""
        text_parts = []
        
        for item in data:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text_parts.append(self._extract_text_from_dict(item))
            else:
                text_parts.append(str(item))
        
        return '\n'.join(text_parts)
    
    def _chunk_text_content(self, text: str, file_path: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对文本内容进行分块"""
        # 清理文本
        cleaned_text = TextUtils.clean_text(text)

        if not cleaned_text.strip():
            logger.warning(f"No text content found in {file_path}")
            return []
        
        # 如果source_info为None，创建默认的source_info
        if source_info is None:
            source_info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_hash': FileUtils.get_file_hash(file_path) if hasattr(FileUtils, 'get_file_hash') else 'unknown'
            }
        
        # 事件链感知的分块处理
        if self.event_chain_optimization:
            text_chunks = self._event_aware_chunking(cleaned_text)
        else:
            # 使用TextUtils进行分块
            text_chunks = TextUtils.chunk_text(
                cleaned_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap
            )

        return self._build_chunks_from_text(cleaned_text, text_chunks, source_info)

    def _chunk_paragraph_by_sentence(self, para_text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于句子序列的段内分句分块（无重叠）"""

        cleaned_para = TextUtils.clean_text(para_text)
        if not cleaned_para:
            return []

        sentences = TextUtils.split_by_sentence(para_text)
        if not sentences:
            sentences = [cleaned_para]

        filtered = [s for s in sentences if len(s.strip()) >= 10]
        if filtered:
            sentences = filtered

        try:
            budget = int(config.get('document.chunk_size', self.chunk_size))
        except Exception:
            budget = self.chunk_size or 512
        if not budget or budget <= 0:
            budget = 512

        chunks_for_para: List[Dict[str, Any]] = []
        cur_sents: List[str] = []
        cur_ids: List[int] = []
        cur_len = 0

        for sid, sentence in enumerate(sentences):
            sentence_clean = TextUtils.clean_text(sentence)
            if not sentence_clean:
                continue

            sentence_len = len(sentence_clean)
            if cur_sents and cur_len + sentence_len > budget:
                chunks_for_para.append({
                    'text': " ".join(cur_sents).strip(),
                    'length': sum(len(s) for s in cur_sents),
                    'sentence_ids': cur_ids.copy(),
                })
                cur_sents = []
                cur_ids = []
                cur_len = 0

            cur_sents.append(sentence_clean)
            cur_ids.append(sid)
            cur_len += sentence_len

        if cur_sents:
            chunks_for_para.append({
                'text': " ".join(cur_sents).strip(),
                'length': sum(len(s) for s in cur_sents),
                'sentence_ids': cur_ids.copy(),
            })

        if not chunks_for_para:
            return []

        return self._build_chunks_from_text(
            cleaned_para,
            chunks_for_para,
            source_info,
            preserve_original_text=True,
        )

    def _build_chunks_from_text(
        self,
        cleaned_text: str,
        text_chunks: List[Dict[str, Any]],
        source_info: Dict[str, Any],
        *,
        preserve_original_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """统一构建 chunk 数据结构，复用上下文与实体提取逻辑"""

        search_pos = 0
        chunks: List[Dict[str, Any]] = []

        for i, chunk_data in enumerate(text_chunks):
            original_text = TextUtils.clean_text(chunk_data.get('text', ''))
            if not original_text:
                continue

            start_idx = cleaned_text.find(original_text, search_pos)
            if start_idx == -1:
                start_idx = search_pos

            pre_context = cleaned_text[max(0, start_idx - 100):start_idx]
            pre_entities = TextUtils.extract_entities(pre_context)
            chunk_entities = TextUtils.extract_entities(original_text)
            pronouns = {'He', 'She', 'They', 'It', 'him', 'her', 'them', 'his', 'her', 'their'}
            chunk_entities = [e for e in chunk_entities if e not in pronouns]

            final_text = original_text
            primary_entity = None
            if chunk_entities:
                primary_entity = chunk_entities[0]
            elif pre_entities:
                primary_entity = pre_entities[-1]
                if not preserve_original_text:
                    final_text = f"{primary_entity} {original_text}"

            chunk = {
                'text': final_text,
                'chunk_index': i,
                'chunk_id': f"{source_info.get('file_name', 'unknown')}_{i:04d}",
                'length': chunk_data.get('length', len(original_text)),
                'source_info': source_info.copy(),
                'created_at': self._get_timestamp(),
                'primary_entity': primary_entity
            }

            sentence_ids = chunk_data.get('sentence_ids')
            if sentence_ids is not None:
                chunk['sentence_ids'] = list(sentence_ids)

            chunk['context'] = self._extract_context_info(final_text, cleaned_text, i)

            search_pos = start_idx + len(original_text)
            chunks.append(chunk)

        return chunks
    
    def _event_aware_chunking(self, text: str) -> List[Dict[str, Any]]:
        """事件链感知的文档分块"""
        import re
        
        # 首先进行标准分块
        standard_chunks = TextUtils.chunk_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        # 识别包含事件关键词的句子
        event_sentences = self._identify_event_sentences(text)
        
        if not event_sentences:
            return standard_chunks
        
        # 优化分块边界，确保事件链完整性
        optimized_chunks = self._optimize_chunk_boundaries(standard_chunks, event_sentences, text)
        
        return optimized_chunks
    
    def _identify_event_sentences(self, text: str) -> List[Dict[str, Any]]:
        """识别包含事件关键词的句子"""
        import re
        
        sentences = re.split(r'[。！？.!?]', text)
        event_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sentence_events = []
            for event_type, keywords in self.event_keywords.items():
                for keyword in keywords:
                    if keyword in sentence:
                        sentence_events.append(event_type)
                        break
            
            if sentence_events:
                # 计算句子在原文中的位置
                sentence_start = text.find(sentence.strip())
                event_sentences.append({
                    'text': sentence.strip(),
                    'index': i,
                    'start_pos': sentence_start,
                    'end_pos': sentence_start + len(sentence.strip()),
                    'event_types': sentence_events
                })
        
        return event_sentences
    
    def _optimize_chunk_boundaries(self, chunks: List[Dict[str, Any]], 
                                  event_sentences: List[Dict[str, Any]], 
                                  full_text: str) -> List[Dict[str, Any]]:
        """优化分块边界以保持事件链完整性"""
        if not event_sentences:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_start = full_text.find(chunk_text)
            chunk_end = chunk_start + len(chunk_text)
            
            # 检查当前块是否包含事件句子
            chunk_events = []
            for event_sent in event_sentences:
                if (event_sent['start_pos'] >= chunk_start and 
                    event_sent['end_pos'] <= chunk_end):
                    chunk_events.append(event_sent)
            
            if not chunk_events:
                # 没有事件句子，保持原样
                optimized_chunks.append(chunk)
                continue
            
            # 检查是否需要扩展块以包含完整的事件链
            extended_chunk = self._extend_chunk_for_event_chain(
                chunk, chunk_events, event_sentences, full_text
            )
            
            # 检查扩展后的块是否过大
            if len(extended_chunk['text']) > self.chunk_size * 1.5:
                # 如果过大，尝试分割但保持事件完整性
                split_chunks = self._split_large_event_chunk(extended_chunk, chunk_events)
                optimized_chunks.extend(split_chunks)
            else:
                optimized_chunks.append(extended_chunk)
        
        # 去重和合并相似的块
        final_chunks = self._merge_overlapping_chunks(optimized_chunks)
        
        return final_chunks
    
    def _extend_chunk_for_event_chain(self, chunk: Dict[str, Any], 
                                     chunk_events: List[Dict[str, Any]], 
                                     all_events: List[Dict[str, Any]], 
                                     full_text: str) -> Dict[str, Any]:
        """扩展块以包含完整的事件链"""
        chunk_text = chunk['text']
        
        # 查找相关的事件句子（前后文中的相关事件）
        related_events = self._find_related_events(chunk_events, all_events)
        
        if not related_events:
            return chunk
        
        # 计算需要扩展的范围
        min_start = min(event['start_pos'] for event in related_events)
        max_end = max(event['end_pos'] for event in related_events)
        
        # 扩展到句子边界
        extended_start = max(0, min_start - 50)  # 向前扩展50字符
        extended_end = min(len(full_text), max_end + 50)  # 向后扩展50字符
        
        # 调整到句子边界
        import re
        sentence_starts = [m.start() for m in re.finditer(r'[。！？.!?]\s*', full_text)]
        
        # 找到最近的句子开始位置
        for start_pos in reversed(sentence_starts):
            if start_pos <= extended_start:
                extended_start = start_pos + 1
                break
        
        # 找到最近的句子结束位置
        for end_pos in sentence_starts:
            if end_pos >= extended_end:
                extended_end = end_pos + 1
                break
        
        extended_text = full_text[extended_start:extended_end].strip()
        
        # 创建扩展后的块
        extended_chunk = chunk.copy()
        extended_chunk['text'] = extended_text
        extended_chunk['length'] = len(extended_text)
        extended_chunk['event_chain_optimized'] = True
        extended_chunk['related_events'] = [event['event_types'] for event in related_events]
        
        return extended_chunk
    
    def _find_related_events(self, chunk_events: List[Dict[str, Any]], 
                           all_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找相关的事件句子"""
        if not chunk_events:
            return []
        
        related_events = chunk_events.copy()
        
        # 查找时间上相近的事件
        for chunk_event in chunk_events:
            chunk_pos = chunk_event['start_pos']
            
            for other_event in all_events:
                if other_event in related_events:
                    continue
                
                # 检查距离（在500字符范围内）
                distance = abs(other_event['start_pos'] - chunk_pos)
                if distance <= 500:
                    # 检查事件类型相关性
                    if self._are_events_related(chunk_event['event_types'], other_event['event_types']):
                        related_events.append(other_event)
        
        return related_events
    
    def _are_events_related(self, events1: List[str], events2: List[str]) -> bool:
        """判断两组事件是否相关"""
        # 定义事件关联规则
        related_pairs = {
            ('acquisition', 'ownership'),
            ('succession', 'ownership'),
            ('merger', 'acquisition'),
            ('bankruptcy', 'acquisition'),
            ('partnership', 'merger')
        }
        
        for event1 in events1:
            for event2 in events2:
                if event1 == event2:  # 同类事件
                    return True
                if (event1, event2) in related_pairs or (event2, event1) in related_pairs:
                    return True
        
        return False
    
    def _split_large_event_chunk(self, chunk: Dict[str, Any], 
                                events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分割过大的事件块，但保持事件完整性"""
        chunk_text = chunk['text']
        
        if len(chunk_text) <= self.chunk_size * 1.2:
            return [chunk]
        
        # 按事件分组
        event_groups = self._group_events_by_proximity(events)
        
        split_chunks = []
        current_pos = 0
        
        for group in event_groups:
            group_start = min(event['start_pos'] for event in group) - current_pos
            group_end = max(event['end_pos'] for event in group) - current_pos
            
            # 确保不超出当前块范围
            group_start = max(0, group_start)
            group_end = min(len(chunk_text), group_end)
            
            if group_end > group_start:
                group_text = chunk_text[group_start:group_end]
                
                split_chunk = chunk.copy()
                split_chunk['text'] = group_text
                split_chunk['length'] = len(group_text)
                split_chunk['event_group'] = [event['event_types'] for event in group]
                
                split_chunks.append(split_chunk)
        
        return split_chunks if split_chunks else [chunk]
    
    def _group_events_by_proximity(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """按接近程度对事件分组"""
        if not events:
            return []
        
        # 按位置排序
        sorted_events = sorted(events, key=lambda x: x['start_pos'])
        
        groups = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            # 如果与当前组的最后一个事件距离小于200字符，加入当前组
            if event['start_pos'] - current_group[-1]['end_pos'] <= 200:
                current_group.append(event)
            else:
                # 否则开始新组
                groups.append(current_group)
                current_group = [event]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _merge_overlapping_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并重叠的块"""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # 检查重叠
            overlap_ratio = self._calculate_text_overlap(current_chunk['text'], next_chunk['text'])
            
            if overlap_ratio > 0.7:  # 70%以上重叠则合并
                # 合并块
                merged_text = self._merge_chunk_texts(current_chunk['text'], next_chunk['text'])
                current_chunk['text'] = merged_text
                current_chunk['length'] = len(merged_text)
                
                # 合并事件信息
                if 'related_events' in current_chunk and 'related_events' in next_chunk:
                    current_chunk['related_events'].extend(next_chunk['related_events'])
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的重叠比例"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的重叠检测：检查较短文本在较长文本中的出现
        shorter, longer = (text1, text2) if len(text1) < len(text2) else (text2, text1)
        
        if shorter in longer:
            return len(shorter) / len(longer)
        
        # 检查部分重叠
        max_overlap = 0
        for i in range(len(shorter)):
            for j in range(i + 1, len(shorter) + 1):
                substring = shorter[i:j]
                if len(substring) > 10 and substring in longer:  # 至少10字符的重叠
                    max_overlap = max(max_overlap, len(substring))
        
        return max_overlap / len(shorter) if len(shorter) > 0 else 0.0
    
    def _merge_chunk_texts(self, text1: str, text2: str) -> str:
        """智能合并两个文本块"""
        # 找到最佳合并点
        overlap_start = -1
        max_overlap_len = 0
        
        # 查找text1结尾和text2开头的重叠
        for i in range(min(100, len(text1))):
            suffix = text1[-(i+1):]
            if text2.startswith(suffix):
                if len(suffix) > max_overlap_len:
                    max_overlap_len = len(suffix)
                    overlap_start = len(text1) - len(suffix)
        
        if overlap_start >= 0:
            # 有重叠，去除重复部分
            return text1[:overlap_start] + text2
        else:
            # 无重叠，直接连接
            return text1 + " " + text2
    
    def _extract_context_info(self, chunk_text: str, full_text: str, chunk_index: int) -> Dict[str, Any]:
        """提取块的上下文信息"""
        context = {
            'position_ratio': self._calculate_position_ratio(chunk_text, full_text),
            'chunk_index': chunk_index,
            'is_beginning': chunk_index == 0,
            'contains_title': self._contains_title_markers(chunk_text),
            'paragraph_count': chunk_text.count('\n\n') + 1
        }
        
        return context
    
    def _calculate_position_ratio(self, chunk_text: str, full_text: str) -> float:
        """计算块在文档中的位置比例"""
        try:
            chunk_start = full_text.find(chunk_text)
            if chunk_start == -1:
                return 0.0
            
            return chunk_start / len(full_text)
        except Exception:
            return 0.0
    
    def _contains_title_markers(self, text: str) -> bool:
        """检查文本是否包含标题标记"""
        title_markers = ['#', '##', '###', '第', '章', '节', '一、', '二、', '三、', '1.', '2.', '3.']
        
        for marker in title_markers:
            if marker in text:
                return True
        
        return False
    
    def _extract_paragraph_idx_mapping(self, data: Dict[str, Any]) -> Dict[str, int]:
        """从musique数据中提取段落文本到idx的映射"""
        mapping: Dict[str, int] = {}
        paragraphs = data.get('paragraphs', [])

        for para in paragraphs:
            if isinstance(para, dict) and 'paragraph_text' in para and 'idx' in para:
                original_text = para['paragraph_text']
                cleaned_text = TextUtils.clean_text(original_text)
                idx = para['idx']

                # 使用清理后的段落文本的前100个字符作为key，避免完全匹配的问题
                key = cleaned_text[:100] if len(cleaned_text) > 100 else cleaned_text
                mapping[key] = idx
                # 同时保存完整清理后的文本作为备用
                mapping[cleaned_text] = idx

                # 保留原始文本以便诊断
                self.paragraph_original_texts[idx] = original_text

        return mapping
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def adaptive_chunking(self, text: str, target_chunk_count: int = None) -> List[Dict[str, Any]]:
        """自适应分块，根据文本长度调整块大小"""
        text_length = len(text)
        
        if target_chunk_count:
            # 根据目标块数量调整块大小
            adaptive_chunk_size = max(100, text_length // target_chunk_count)
        else:
            # 根据文本长度自适应
            if text_length < 1000:
                adaptive_chunk_size = text_length
            elif text_length < 5000:
                adaptive_chunk_size = 500
            elif text_length < 20000:
                adaptive_chunk_size = 1000
            else:
                adaptive_chunk_size = 1500
        
        # 临时调整配置
        original_chunk_size = self.chunk_size
        self.chunk_size = adaptive_chunk_size
        
        try:
            chunks = TextUtils.chunk_text(text, chunk_size=adaptive_chunk_size, overlap=self.overlap)
            return chunks
        finally:
            # 恢复原始配置
            self.chunk_size = original_chunk_size
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证分块结果"""
        valid_chunks = []
        
        for chunk in chunks:
            # 检查基本字段
            if not chunk.get('text') or len(chunk['text'].strip()) < 10:
                logger.warning(f"Skipping chunk with insufficient content: {chunk.get('chunk_id')}")
                continue
            
            # 检查文本质量
            text = chunk['text']
            if self._is_low_quality_text(text):
                logger.warning(f"Skipping low quality chunk: {chunk.get('chunk_id')}")
                continue
            
            valid_chunks.append(chunk)
        
        logger.info(f"Validated {len(valid_chunks)} out of {len(chunks)} chunks")
        return valid_chunks
    
    def _is_low_quality_text(self, text: str) -> bool:
        """检查文本质量"""
        # 检查是否主要由特殊字符组成
        import string
        
        # 计算字母数字字符的比例
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        total_count = len(text)
        
        if total_count == 0:
            return True
        
        alphanumeric_ratio = alphanumeric_count / total_count
        
        # 如果字母数字字符比例太低，认为是低质量文本
        return alphanumeric_ratio < 0.3
