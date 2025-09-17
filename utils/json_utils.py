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

    # Step 1: 剥离 markdown 代码块
    text = _strip_markdown_code_blocks(text)

    # If the whole text is JSON
    try:
        parsed = json.loads(text)
        # Step 2: 容忍外层对象，自动提取数组字段
        extracted = _extract_array_from_wrapper(parsed)
        if extracted is not None:
            return json.dumps(extracted, ensure_ascii=False)
        return text
    except Exception:
        pass

    # Search for JSON object or array in the text
    brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if brace_match:
        candidate = clean_control_characters(brace_match.group(1))
        try:
            parsed = json.loads(candidate)
            # Step 2: 容忍外层对象，自动提取数组字段
            extracted = _extract_array_from_wrapper(parsed)
            if extracted is not None:
                return json.dumps(extracted, ensure_ascii=False)
            return candidate
        except Exception:
            # 尝试修复常见的JSON格式问题
            fixed_candidate = _try_fix_json_format(candidate)
            if fixed_candidate:
                try:
                    parsed = json.loads(fixed_candidate)
                    # Step 2: 容忍外层对象，自动提取数组字段
                    extracted = _extract_array_from_wrapper(parsed)
                    if extracted is not None:
                        return json.dumps(extracted, ensure_ascii=False)
                    return fixed_candidate
                except Exception:
                    pass

    # 尝试从文本中提取关键信息构建JSON
    fallback_json = _extract_fallback_json(text)
    if fallback_json:
        return fallback_json

    return ""


def _strip_markdown_code_blocks(text: str) -> str:
    """剥离 markdown 代码块，支持多种格式"""
    if not text:
        return text
    
    # 处理完整的代码块 ```json ... ``` 或 ``` ... ```
    code_block_pattern = r'```(?:json|JSON)?\s*(.*?)\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 处理单行代码块 `...`
    if text.startswith('`') and text.endswith('`') and text.count('`') == 2:
        return text[1:-1].strip()
    
    return text


def _extract_array_from_wrapper(parsed_data) -> list:
    """从外层对象中提取数组字段，容忍 {"data": [...]} 或 {"result": [...]} 格式"""
    if isinstance(parsed_data, list):
        return parsed_data
    
    if isinstance(parsed_data, dict):
        # 检查常见的包装字段
        for key in ['data', 'result', 'results', 'items', 'content', 'facts', 'sentences']:
            if key in parsed_data and isinstance(parsed_data[key], list):
                return parsed_data[key]
    
    return None


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
        
        # 不再使用summary字段
        
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
            "keywords": keywords,
            "entities": entities,
            "concepts": [],
            "importance_score": 0.5,
            "note_type": "fact"
        }
        
        return json.dumps(fallback_data, ensure_ascii=False)
    
    except Exception:
        return ""
