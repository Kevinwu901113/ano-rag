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
        
    def chunk_document(self, file_path: str, source_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """对单个文档进行分块"""
        logger.info(f"Chunking document: {file_path}")
        
        try:
            # 读取文档内容
            content = self._read_document_content(file_path)
            
            # 特殊处理：如果是musique格式的JSONL数据，需要提取idx信息
            paragraph_idx_mapping = None
            
            # 处理JSONL格式：如果content是列表，取第一个元素
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if isinstance(first_item, dict) and 'paragraphs' in first_item:
                    paragraph_idx_mapping = self._extract_paragraph_idx_mapping(first_item)
                    # 使用第一个JSONL条目作为内容
                    content = first_item
            elif isinstance(content, dict) and 'paragraphs' in content:
                paragraph_idx_mapping = self._extract_paragraph_idx_mapping(content)
            
            # 提取文本内容
            text_content = self._extract_text_content(content, file_path)
            
            # 分块处理
            chunks = self._chunk_text_content(text_content, file_path, source_info)
            
            # 为每个chunk添加paragraph_idx_mapping信息
            if paragraph_idx_mapping:
                for chunk in chunks:
                    chunk['paragraph_idx_mapping'] = paragraph_idx_mapping
            
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
        
        # 处理musique数据集格式：包含paragraphs和question字段
        if 'paragraphs' in data and 'question' in data:
            # 提取所有段落文本
            paragraphs = data.get('paragraphs', [])
            for para in paragraphs:
                if isinstance(para, dict) and 'paragraph_text' in para:
                    text_parts.append(para['paragraph_text'])
            
            # 添加问题
            question = data.get('question', '')
            if question:
                text_parts.append(f"Question: {question}")
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
        
        # 使用TextUtils进行分块
        text_chunks = TextUtils.chunk_text(
            cleaned_text,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        search_pos = 0  # 用于在原文中定位每个块的起始位置
        
        # 创建分块数据结构
        chunks = []
        for i, chunk_data in enumerate(text_chunks):
            original_text = chunk_data['text']

            # 在全文中定位当前块的起始位置
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
            if not chunk_entities and pre_entities:
                primary_entity = pre_entities[-1]
                final_text = f"{primary_entity} {original_text}"
            elif chunk_entities:
                primary_entity = chunk_entities[0]

            chunk = {
                'text': final_text,
                'chunk_index': i,
                'chunk_id': f"{source_info.get('file_name', 'unknown')}_{i:04d}",
                'length': chunk_data['length'],
                'source_info': source_info.copy(),
                'created_at': self._get_timestamp(),
                'primary_entity': primary_entity
            }

            # 添加上下文信息
            chunk['context'] = self._extract_context_info(final_text, cleaned_text, i)

            search_pos = start_idx + len(original_text)

            chunks.append(chunk)
        
        return chunks
    
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
        mapping = {}
        paragraphs = data.get('paragraphs', [])
        
        for para in paragraphs:
            if isinstance(para, dict) and 'paragraph_text' in para and 'idx' in para:
                paragraph_text = para['paragraph_text']
                idx = para['idx']
                # 使用段落文本的前100个字符作为key，避免完全匹配的问题
                key = paragraph_text[:100] if len(paragraph_text) > 100 else paragraph_text
                mapping[key] = idx
                # 同时保存完整文本作为备用
                mapping[paragraph_text] = idx
        
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