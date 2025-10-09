"""
统一的原子笔记响应解析器
用于解析LLM生成的原子笔记响应，支持哨兵字符和容错处理
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

from config import config
from utils.note_completeness import is_complete_sentence

_REL_LEX: Optional[Dict[str, List[re.Pattern]]] = None
_TYPE_HINTS: Optional[Dict[str, List[str]]] = None
_NORM: Optional[Dict[str, Any]] = None
_DEFAULT_REL: str = "related_to"


def _load_lex() -> None:
    """Lazy-load lexical resources for relation and type inference."""

    global _REL_LEX, _TYPE_HINTS, _NORM, _DEFAULT_REL
    note_key_cfg = config.get("note_keys", {}) or {}

    if _REL_LEX is None:
        lex = note_key_cfg.get("rel_lexicon", {}) or {}
        compiled: Dict[str, List[re.Pattern]] = {}
        for rel, pats in lex.items():
            compiled[rel] = [
                re.compile(r"\b" + re.escape(pat) + r"\b", re.IGNORECASE)
                if not re.search(r"[\u4e00-\u9fff]", str(pat))
                else re.compile(re.escape(str(pat)))
                for pat in (pats or [])
                if pat
            ]
        _REL_LEX = compiled

    if _TYPE_HINTS is None:
        _TYPE_HINTS = note_key_cfg.get("type_hints", {}) or {}

    if _NORM is None:
        _NORM = note_key_cfg.get(
            "normalize",
            {"strip_quotes": True, "collapse_space": True, "lower": False},
        ) or {}

    default_rel = note_key_cfg.get("default_rel")
    if isinstance(default_rel, str) and default_rel.strip():
        _DEFAULT_REL = default_rel.strip()


def _norm(value: str) -> str:
    if not value:
        return value

    result = value
    if _NORM.get("strip_quotes", True):
        result = result.strip().strip("\"'“”‘’")
    if _NORM.get("collapse_space", True):
        result = re.sub(r"\s+", " ", result).strip()
    if _NORM.get("lower", False):
        result = result.lower()
    return result


def _extract_rel(text: str) -> str:
    _load_lex()
    for rel, patterns in (_REL_LEX or {}).items():
        if any(pat.search(text) for pat in patterns):
            return rel
    return _DEFAULT_REL


def _split_by_rel(text: str, rel: str) -> Tuple[str, str]:
    _load_lex()
    for pat in (_REL_LEX or {}).get(rel, []):
        match = pat.search(text)
        if match:
            left = text[: match.start()].strip()
            right = text[match.end() :].strip()
            return left, right

    fallback_tokens = config.get("note_keys.fallback_splitters", []) or []
    for token in fallback_tokens:
        pattern = re.compile(re.escape(str(token)), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[: match.start()].strip(), text[match.end() :].strip()

    match = re.search(r"\b(is|为|由|是由|是|in|of|by)\b", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip(), text[match.end() :].strip()

    return text, ""


def _normalize_source_sent_ids_field(notes: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for note in notes or []:
        if not isinstance(note, dict):
            continue

        raw_ids = note.get('source_sent_ids', [])
        if isinstance(raw_ids, int):
            raw_ids = [raw_ids]
        elif not isinstance(raw_ids, (list, tuple, set)):
            raw_ids = []

        cleaned: List[int] = []
        for value in raw_ids:
            try:
                cleaned.append(int(str(value).strip()))
            except Exception:
                continue

        note['source_sent_ids'] = sorted(set(cleaned))
        normalized.append(note)

    return normalized


def _infer_type(literal: str, rel: str) -> str:
    literal_lower = (literal or "").lower()
    if literal_lower:
        for entity_type, hints in (_TYPE_HINTS or {}).items():
            for hint in hints or []:
                if str(hint).lower() in literal_lower:
                    return entity_type

    if rel in ("performed_by", "composed_by", "directed_by"):
        return "album"
    if rel in ("spouse_of", "partner_of", "born_in"):
        return "person"
    return ""


def enrich_note_keys(note: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically backfill relation and literal keys from text."""

    if not isinstance(note, dict):
        return note

    text = str(note.get("text") or "").strip()
    if not text:
        return note

    _load_lex()
    rel = str(note.get("rel") or "").strip() or _extract_rel(text)
    if rel not in (_REL_LEX or {}):
        rel = _extract_rel(text)

    head_key = note.get("head_key") or ""
    tail_key = note.get("tail_key") or ""

    if not (head_key and tail_key):
        left, right = _split_by_rel(text, rel)
        head_key = head_key or _norm(left)
        tail_key = tail_key or _norm(right)

    type_head = note.get("type_head") or _infer_type(head_key, rel)
    type_tail = note.get("type_tail") or _infer_type(tail_key, rel)

    note.update(
        {
            "rel": rel,
            "head_key": head_key,
            "tail_key": tail_key,
            "type_head": type_head,
            "type_tail": type_tail,
        }
    )
    return note


def parse_notes_response(raw: str, sentinel: str = "~") -> Optional[List[Dict[str, Any]]]:
    """
    统一解析器，解析LLM生成的原子笔记响应
    按照指定顺序进行解析，增强容错处理
    
    Args:
        raw: LLM的原始响应文本
        sentinel: 哨兵字符，表示0条笔记
        
    Returns:
        List[Dict]: 解析成功返回笔记列表，失败返回None触发重试/回退
    """
    if not raw:
        return []
    
    s = raw.strip()
    
    # 1. 哨兵字符检查：0条笔记
    if s == sentinel:
        logger.debug(f"Detected sentinel character '{sentinel}', returning empty list")
        return []
    
    # 2. 最短JSON检查：0条笔记
    if s == "[]":
        logger.debug("Detected empty JSON array, returning empty list")
        return []
    
    # 3. 尝试标准JSON解析
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            logger.debug(f"Successfully parsed JSON array with {len(obj)} items")
            return _normalize_source_sent_ids_field(obj)
        elif isinstance(obj, dict):
            # 如果是单个对象，包装成列表
            logger.debug("Parsed single JSON object, wrapping in list")
            return _normalize_source_sent_ids_field([obj])
    except json.JSONDecodeError as e:
        logger.debug(f"Standard JSON parsing failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during JSON parsing: {e}")
    
    # 4. 从文本尾部提取最外层JSON数组（兜住"模型先讲两句解释、最后才输出数组"的情况）
    json_match = re.search(r'\[[\s\S]*\]$', s)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, list):
                logger.debug(f"Successfully extracted JSON array from text tail with {len(obj)} items")
                return _normalize_source_sent_ids_field(obj)
        except json.JSONDecodeError as e:
            logger.debug(f"Extracted JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during extracted JSON parsing: {e}")
    
    # 5. 容错：尝试提取单个JSON对象
    json_obj_match = re.search(r'\{[\s\S]*\}', s)
    if json_obj_match:
        try:
            obj = json.loads(json_obj_match.group(0))
            if isinstance(obj, dict):
                logger.debug("Successfully extracted single JSON object from text")
                return _normalize_source_sent_ids_field([obj])
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

        text = str(note.get('text', '')).strip()
        if not is_complete_sentence(text, None):
            logger.debug(f"Reject incomplete/fragment note: {text}")
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
