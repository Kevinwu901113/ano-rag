"""
统一的原子笔记响应解析器
用于解析LLM生成的原子笔记响应，支持哨兵字符和容错处理
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from loguru import logger


def parse_notes_response(raw: str, sentinel: str = "~") -> Optional[List[Dict[str, Any]]]:
    """
    统一解析器，解析LLM生成的原子笔记响应
    
    Args:
        raw: LLM的原始响应文本
        sentinel: 哨兵字符，表示0条笔记
        
    Returns:
        List[Dict]: 解析成功返回笔记列表，失败返回None触发重试/回退
    """
    if not raw:
        return []
    
    s = raw.strip()
    
    # 哨兵字符检查：0条笔记
    if s == sentinel:
        logger.debug(f"Detected sentinel character '{sentinel}', returning empty list")
        return []
    
    # 最短JSON检查：0条笔记
    if s == "[]":
        logger.debug("Detected empty JSON array, returning empty list")
        return []
    
    # 尝试标准JSON解析
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            logger.debug(f"Successfully parsed JSON array with {len(obj)} items")
            return obj
        elif isinstance(obj, dict):
            # 如果是单个对象，包装成列表
            logger.debug("Parsed single JSON object, wrapping in list")
            return [obj]
    except json.JSONDecodeError as e:
        logger.debug(f"Standard JSON parsing failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during JSON parsing: {e}")
    
    # 容错：从文本中提取JSON数组
    json_match = re.search(r'\[[\s\S]*\]', s)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, list):
                logger.debug(f"Successfully extracted JSON array from text with {len(obj)} items")
                return obj
        except json.JSONDecodeError as e:
            logger.debug(f"Extracted JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during extracted JSON parsing: {e}")
    
    # 容错：尝试提取单个JSON对象
    json_obj_match = re.search(r'\{[\s\S]*\}', s)
    if json_obj_match:
        try:
            obj = json.loads(json_obj_match.group(0))
            if isinstance(obj, dict):
                logger.debug("Successfully extracted single JSON object from text")
                return [obj]
        except json.JSONDecodeError as e:
            logger.debug(f"Extracted JSON object parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during extracted JSON object parsing: {e}")
    
    # 所有解析方法都失败，返回None触发重试/回退
    logger.warning(f"All parsing methods failed for response: {s[:200]}...")
    return None


def validate_note_structure(note: Dict[str, Any]) -> bool:
    """
    验证单个笔记的结构是否符合要求
    
    Args:
        note: 笔记字典
        
    Returns:
        bool: 结构有效返回True，否则返回False
    """
    if not isinstance(note, dict):
        return False
    
    # 必需字段检查
    required_fields = ['text', 'sent_count', 'salience']
    for field in required_fields:
        if field not in note:
            logger.debug(f"Missing required field: {field}")
            return False
    
    # 字段类型检查
    try:
        # text应该是字符串
        if not isinstance(note['text'], str) or not note['text'].strip():
            logger.debug("Invalid text field")
            return False
        
        # sent_count应该是正整数
        sent_count = note['sent_count']
        if not isinstance(sent_count, int) or sent_count < 1:
            logger.debug(f"Invalid sent_count: {sent_count}")
            return False
        
        # salience应该是0-1之间的数值
        salience = note['salience']
        if not isinstance(salience, (int, float)) or not (0 <= salience <= 1):
            logger.debug(f"Invalid salience: {salience}")
            return False
        
        return True
        
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"Structure validation error: {e}")
        return False


def filter_valid_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤出结构有效的笔记
    
    Args:
        notes: 笔记列表
        
    Returns:
        List[Dict]: 有效的笔记列表
    """
    if not isinstance(notes, list):
        return []
    
    valid_notes = []
    for i, note in enumerate(notes):
        if validate_note_structure(note):
            valid_notes.append(note)
        else:
            logger.debug(f"Filtered out invalid note at index {i}")
    
    logger.debug(f"Filtered {len(valid_notes)} valid notes from {len(notes)} total")
    return valid_notes


def normalize_note_fields(note: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化笔记字段，确保字段类型和格式正确
    
    Args:
        note: 原始笔记字典
        
    Returns:
        Dict: 标准化后的笔记字典
    """
    normalized = note.copy()
    
    # 确保text字段是字符串
    if 'text' in normalized:
        normalized['text'] = str(normalized['text']).strip()
    
    # 确保sent_count是整数
    if 'sent_count' in normalized:
        try:
            normalized['sent_count'] = int(normalized['sent_count'])
        except (ValueError, TypeError):
            normalized['sent_count'] = 1
    
    # 确保salience是浮点数
    if 'salience' in normalized:
        try:
            normalized['salience'] = float(normalized['salience'])
            # 限制在0-1范围内
            normalized['salience'] = max(0.0, min(1.0, normalized['salience']))
        except (ValueError, TypeError):
            normalized['salience'] = 0.5
    
    # 确保列表字段是列表
    list_fields = ['local_spans', 'entities', 'years', 'quality_flags']
    for field in list_fields:
        if field in normalized:
            if not isinstance(normalized[field], list):
                if isinstance(normalized[field], str):
                    # 尝试解析字符串为列表
                    try:
                        normalized[field] = json.loads(normalized[field])
                        if not isinstance(normalized[field], list):
                            normalized[field] = [normalized[field]]
                    except:
                        normalized[field] = [normalized[field]]
                else:
                    normalized[field] = [normalized[field]] if normalized[field] is not None else []
        else:
            normalized[field] = []
    
    return normalized