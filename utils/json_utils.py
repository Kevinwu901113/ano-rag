import json
import re


def clean_control_characters(text: str) -> str:
    """Remove most control characters but keep common whitespace."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    cleaned = cleaned.replace('\u0000', '').replace('\u0001', '').replace('\u0002', '')
    return cleaned


def extract_json_from_response(response: str) -> str:
    """Try to extract a JSON string from a larger LLM response."""
    if not response or not str(response).strip():
        return ""

    text = clean_control_characters(str(response).strip())

    # If the whole text is JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Remove common markdown code block markers
    if text.startswith('```') and text.endswith('```'):
        text_block = re.sub(r'^```(?:json)?', '', text[:-3], flags=re.IGNORECASE).strip()
        try:
            json.loads(text_block)
            return text_block
        except Exception:
            pass

    # Search for JSON inside markdown code block
    code_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if code_match:
        candidate = clean_control_characters(code_match.group(1).strip())
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # Search for JSON object or array in the text
    brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if brace_match:
        candidate = clean_control_characters(brace_match.group(1))
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # 尝试修复常见的JSON格式问题
            fixed_candidate = _try_fix_json_format(candidate)
            if fixed_candidate:
                try:
                    json.loads(fixed_candidate)
                    return fixed_candidate
                except Exception:
                    pass

    # 尝试从文本中提取关键信息构建JSON
    fallback_json = _extract_fallback_json(text)
    if fallback_json:
        return fallback_json

    return ""


def _try_fix_json_format(json_str: str) -> str:
    """尝试修复常见的JSON格式问题"""
    if not json_str:
        return ""
    
    # 移除尾随的逗号
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # 修复未闭合的引号
    # 简单的启发式方法：如果引号数量是奇数，在末尾添加引号
    quote_count = json_str.count('"') - json_str.count('\\"')
    if quote_count % 2 == 1:
        json_str += '"'
    
    # 修复未闭合的大括号
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    # 修复未闭合的方括号
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    return json_str


def _extract_fallback_json(text: str) -> str:
    """从文本中提取关键信息构建备用JSON"""
    try:
        # 尝试提取content字段 - 使用更简单的模式
        content_match = re.search(r'content[^:]*:[^"]*"([^"]+)"', text, re.IGNORECASE)
        content = content_match.group(1) if content_match else text[:200]
        
        # 尝试提取summary字段
        summary_match = re.search(r'summary[^:]*:[^"]*"([^"]+)"', text, re.IGNORECASE)
        summary = summary_match.group(1) if summary_match else content[:100]
        
        # 尝试提取keywords字段
        keywords_match = re.search(r'keywords[^:]*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
        keywords = []
        if keywords_match:
            keywords_str = keywords_match.group(1)
            # 简单分割并清理
            keywords = [k.strip(' "\',') for k in keywords_str.split(',') if k.strip()]
        
        # 尝试提取entities字段
        entities_match = re.search(r'entities[^:]*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
        entities = []
        if entities_match:
            entities_str = entities_match.group(1)
            # 简单分割并清理
            entities = [e.strip(' "\',') for e in entities_str.split(',') if e.strip()]
        
        # 构建备用JSON
        fallback_data = {
            "content": content,
            "summary": summary,
            "keywords": keywords,
            "entities": entities,
            "concepts": [],
            "importance_score": 0.5,
            "note_type": "fact"
        }
        
        return json.dumps(fallback_data, ensure_ascii=False)
    
    except Exception:
        return ""
