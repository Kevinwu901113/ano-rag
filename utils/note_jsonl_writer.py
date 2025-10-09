"""
JSONL格式笔记记录器
用于在项目运行时记录生成的所有原子笔记到note.jsonl文件
"""

import json
import os
import threading
from typing import Dict, Any, List, Optional
from loguru import logger
from config import config


class NoteJSONLWriter:
    """原子笔记JSONL文件写入器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化JSONL写入器
        
        Args:
            output_dir: 输出目录，默认使用配置中的工作目录
        """
        self.output_dir = output_dir or config.get('storage.work_dir', '.')
        self.jsonl_file_path = os.path.join(self.output_dir, 'note.jsonl')
        self._lock = threading.Lock()
        self._ensure_output_dir()
        
        # 初始化时清空文件（如果存在）
        self._initialize_file()
        
        logger.info(f"NoteJSONLWriter initialized, output file: {self.jsonl_file_path}")
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _initialize_file(self):
        """初始化JSONL文件"""
        try:
            with open(self.jsonl_file_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
            logger.debug(f"Initialized note.jsonl file: {self.jsonl_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize note.jsonl file: {e}")
    
    def write_note(self, note: Dict[str, Any], question_id: Optional[str] = None, idx: Optional[int] = None):
        """
        写入单个笔记到JSONL文件
        
        Args:
            note: 笔记数据字典
            question_id: 问题ID，如果未提供则尝试从note中获取
            idx: 索引位置，如果未提供则尝试从note中获取
        """
        try:
            # 提取必要字段
            note_id = note.get('note_id', '')
            note_content = note.get('content', note.get('summary', ''))
            
            # 尝试从不同字段获取question_id
            if question_id is None:
                question_id = (note.get('question_id') or 
                             note.get('id') or 
                             note.get('source_info', {}).get('document_id', ''))
            
            # 尝试从不同字段获取idx
            if idx is None:
                idx = (
                    (note.get('paragraph_idxs', [None])[0] if note.get('paragraph_idxs') else None)
                    or note.get('paragraph_idx')
                    or note.get('idx')
                    or note.get('chunk_index')
                    or 0
                )
            
            # 构建JSONL记录
            jsonl_record = {
                'id': str(question_id),
                'note_id': str(note_id),
                'idx': int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else 0,
                'note': str(note_content)
            }
            
            # 线程安全地写入文件
            with self._lock:
                with open(self.jsonl_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
            
            logger.debug(f"Written note to JSONL: {note_id}")
            
        except Exception as e:
            logger.error(f"Failed to write note to JSONL: {e}")
    
    def write_notes_batch(self, notes: List[Dict[str, Any]], question_id: Optional[str] = None):
        """
        批量写入笔记到JSONL文件
        
        Args:
            notes: 笔记列表
            question_id: 问题ID，如果未提供则尝试从每个note中获取
        """
        if not notes:
            return
        
        try:
            records = []
            for i, note in enumerate(notes):
                # 提取必要字段
                note_id = note.get('note_id', '')
                note_content = note.get('content', note.get('summary', ''))
                
                # 尝试从不同字段获取question_id
                current_question_id = question_id
                if current_question_id is None:
                    current_question_id = (note.get('question_id') or 
                                         note.get('id') or 
                                         note.get('source_info', {}).get('document_id', ''))
                
                # 尝试从不同字段获取idx
                idx = (
                    (note.get('paragraph_idxs', [None])[0] if note.get('paragraph_idxs') else None)
                    or note.get('paragraph_idx')
                    or note.get('idx')
                    or note.get('chunk_index')
                    or i
                )
                
                # 构建JSONL记录
                jsonl_record = {
                    'id': str(current_question_id),
                    'note_id': str(note_id),
                    'idx': int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else i,
                    'note': str(note_content)
                }
                records.append(jsonl_record)
            
            # 线程安全地批量写入文件
            with self._lock:
                with open(self.jsonl_file_path, 'a', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Written {len(records)} notes to JSONL file")
            
        except Exception as e:
            logger.error(f"Failed to write notes batch to JSONL: {e}")
    
    def get_file_path(self) -> str:
        """获取JSONL文件路径"""
        return self.jsonl_file_path
    
    def get_record_count(self) -> int:
        """获取当前文件中的记录数量"""
        try:
            with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Failed to count records in JSONL file: {e}")
            return 0
    
    def clear_file(self):
        """清空JSONL文件"""
        try:
            with self._lock:
                with open(self.jsonl_file_path, 'w', encoding='utf-8') as f:
                    pass  # 创建空文件
            logger.info("Cleared note.jsonl file")
        except Exception as e:
            logger.error(f"Failed to clear JSONL file: {e}")


# 全局实例，用于在整个项目中共享
_global_writer: Optional[NoteJSONLWriter] = None
_writer_lock = threading.Lock()


def get_global_note_writer(output_dir: Optional[str] = None) -> NoteJSONLWriter:
    """
    获取全局笔记写入器实例
    
    Args:
        output_dir: 输出目录，仅在首次创建时使用
        
    Returns:
        NoteJSONLWriter实例
    """
    global _global_writer
    
    with _writer_lock:
        # 如果提供了新的输出目录，或者全局写入器不存在，则创建新的
        if _global_writer is None or (output_dir and _global_writer.output_dir != output_dir):
            _global_writer = NoteJSONLWriter(output_dir)
        return _global_writer


def write_note_to_jsonl(note: Dict[str, Any], question_id: Optional[str] = None, idx: Optional[int] = None, output_dir: Optional[str] = None):
    """
    便捷函数：写入单个笔记到全局JSONL文件
    
    Args:
        note: 笔记数据字典
        question_id: 问题ID
        idx: 索引位置
        output_dir: 输出目录
    """
    writer = get_global_note_writer(output_dir)
    writer.write_note(note, question_id, idx)


def write_notes_batch_to_jsonl(notes: List[Dict[str, Any]], question_id: Optional[str] = None, output_dir: Optional[str] = None):
    """
    便捷函数：批量写入笔记到全局JSONL文件
    
    Args:
        notes: 笔记列表
        question_id: 问题ID
        output_dir: 输出目录
    """
    writer = get_global_note_writer(output_dir)
    writer.write_notes_batch(notes, question_id)


def clear_note_jsonl(output_dir: Optional[str] = None):
    """
    便捷函数：清空全局JSONL文件
    
    Args:
        output_dir: 输出目录
    """
    writer = get_global_note_writer(output_dir)
    writer.clear_file()