"""鲁棒的JSON解析器，用于处理LLM输出的各种格式问题"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from loguru import logger

# 禁用的答案短语
FORBIDDEN = {"insufficient information", "no spouse mentioned"}

def extract_json_array(text: str, prefer: str = "array") -> Tuple[Optional[List], Optional[str]]:
    """
    鲁棒地从文本中提取JSON数组或对象
    
    Args:
        text: 输入文本
        prefer: 偏好类型，"array" 优先提取数组，"object" 优先提取对象
        
    Returns:
        (parsed_json, error_message): 成功时返回(解析结果, None)，失败时返回(None, 错误信息)
    """
    if not text or not text.strip():
        return None, "Empty input text"
    
    text = text.strip()
    candidates = []
    
    # 1. 从代码围栏中提取
    code_fence_patterns = [
        r"```(?:json)?\s*(\[.*?\])\s*```",  # JSON数组
        r"```(?:json)?\s*(\{.*?\})\s*```",  # JSON对象
    ]
    
    for pattern in code_fence_patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            candidates.append(match.strip())
    
    # 2. 直接扫描平衡的JSON结构
    # 查找数组 [...]
    array_pattern = r'\[(?:[^\[\]]*(?:\[[^\[\]]*\])*)*[^\[\]]*\]'
    array_matches = re.findall(array_pattern, text, flags=re.DOTALL)
    candidates.extend(array_matches)
    
    # 查找对象 {...}
    # 使用更精确的平衡括号匹配
    brace_candidates = []
    brace_stack = []
    start_pos = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_pos = i
            brace_stack.append(char)
        elif char == '}' and brace_stack:
            brace_stack.pop()
            if not brace_stack and start_pos != -1:
                brace_candidates.append(text[start_pos:i+1])
                start_pos = -1
    
    candidates.extend(brace_candidates)
    
    # 3. 根据prefer参数排序候选项
    def sort_key(candidate):
        is_array = candidate.strip().startswith('[')
        is_object = candidate.strip().startswith('{')
        
        # 优先级：prefer类型 > 长度
        if prefer == "array":
            priority = 0 if is_array else (1 if is_object else 2)
        else:  # prefer == "object"
            priority = 0 if is_object else (1 if is_array else 2)
        
        return (priority, -len(candidate))
    
    candidates.sort(key=sort_key)
    
    # 4. 尝试解析候选项，返回第一个成功的
    errors = []
    
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            # 验证类型匹配
            if prefer == "array" and isinstance(parsed, list):
                return parsed, None
            elif prefer == "object" and isinstance(parsed, dict):
                return parsed, None
            elif prefer == "array" and isinstance(parsed, dict):
                # 如果偏好数组但得到对象，检查是否有包装的数组
                for key, value in parsed.items():
                    if isinstance(value, list):
                        logger.info(f"Found wrapped array in object key '{key}'")
                        return value, None
                return parsed, None  # 返回对象作为备选
            else:
                return parsed, None  # 返回任何有效的JSON
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error in candidate '{candidate[:50]}...': {str(e)}")
            # 尝试修复常见的JSON格式问题
            fixed_candidate = _try_fix_json(candidate.strip())
            if fixed_candidate != candidate.strip():
                try:
                    parsed = json.loads(fixed_candidate)
                    logger.info(f"Successfully fixed and parsed JSON after repair")
                    return parsed, None
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            errors.append(f"Unexpected error in candidate '{candidate[:50]}...': {str(e)}")
    
    # 5. 尝试宽松解析 - 查找任何看起来像JSON的内容
    loose_patterns = [
        r'\[.*?\]',  # 任何方括号内容
        r'\{.*?\}',  # 任何大括号内容
    ]
    
    for pattern in loose_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            if match not in [c.strip() for c in candidates]:  # 避免重复
                fixed_match = _try_fix_json(match.strip())
                try:
                    parsed = json.loads(fixed_match)
                    logger.info(f"Successfully parsed JSON using loose pattern matching")
                    return parsed, None
                except json.JSONDecodeError:
                    continue
    
    # 6. 所有候选项都失败
    if not candidates:
        return None, "No JSON structure found in text"
    
    error_summary = f"Failed to parse {len(candidates)} candidates. Errors: {'; '.join(errors[:3])}"
    return None, error_summary

def _try_fix_json(json_str: str) -> str:
    """
    尝试修复常见的JSON格式问题
    
    Args:
        json_str: 可能有问题的JSON字符串
        
    Returns:
        修复后的JSON字符串
    """
    if not json_str:
        return json_str
    
    # 移除控制字符
    json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
    
    # 修复常见的引号问题
    json_str = json_str.replace("'", '"')  # 单引号改双引号
    
    # 修复尾随逗号
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # 修复缺失的引号（简单情况）
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # 键没有引号
    
    # 修复换行符问题
    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    
    return json_str

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    尝试从raw文本中抽出第一个JSON对象，并解析为dict。
    支持：代码块包裹、前后多余文本、字符串化的JSON。
    
    Args:
        raw: LLM的原始输出文本
        
    Returns:
        解析后的JSON对象
        
    Raises:
        ValueError: 当无法解析出有效的JSON对象时
    """
    if not raw or not raw.strip():
        raise ValueError("Empty input text")
    
    s = raw.strip()
    
    # 1) 去掉```json ... ```包裹
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        s = m.group(1)
    
    # 2) 直接找第一个 {...} JSON 对象
    if not s.startswith("{"):
        m = re.search(r"(\{.*\})", s, flags=re.S)
        if m:
            s = m.group(1)
    
    def try_load(x):
        try:
            return json.loads(x)
        except Exception:
            return None
    
    obj = try_load(s)
    
    # 3) 如果直接解析失败，尝试修复
    if obj is None:
        fixed_s = _try_fix_json(s)
        obj = try_load(fixed_s)
        if obj is not None:
            logger.info("Successfully parsed JSON after applying fixes")
    
    # 4) 处理"字符串化JSON"的情况
    if isinstance(obj, str):
        obj2 = try_load(obj)
        if isinstance(obj2, dict):
            obj = obj2
    
    # 5) 如果还是失败，尝试更宽松的解析
    if not isinstance(obj, dict):
        # 尝试提取任何看起来像JSON对象的内容
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 嵌套对象
            r'\{[^{}]+\}',  # 简单对象
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, raw, re.DOTALL)
            for match in matches:
                fixed_match = _try_fix_json(match.strip())
                test_obj = try_load(fixed_match)
                if isinstance(test_obj, dict):
                    logger.info("Successfully parsed JSON using pattern matching")
                    obj = test_obj
                    break
            if isinstance(obj, dict):
                break
    
    if not isinstance(obj, dict):
        raise ValueError("LLM did not return a valid JSON object")
    
    return obj

def extract_prediction(raw_llm_output: str, passages: Dict[int, str]) -> Tuple[str, List[int]]:
    """
    从LLM输出中提取预测答案和支持段落索引
    
    Args:
        raw_llm_output: LLM的原始输出
        passages: {idx:int -> paragraph_text:str}, 用于轻量校验，不改答案内容
        
    Returns:
        (answer, support_idxs): 答案和支持段落索引列表
        
    Raises:
        ValueError: 当解析失败或答案不符合要求时
    """
    data = parse_llm_json(raw_llm_output)
    
    # 多键尝试：answer → final_answer → prediction → 任一非空字符串字段
    ans = ""
    answer_keys = ["answer", "final_answer", "prediction"]
    
    # 首先尝试预定义的键
    for key in answer_keys:
        if key in data and data[key]:
            ans = str(data[key]).strip()
            if ans:
                break
    
    # 如果预定义键都没有找到答案，尝试任一非空字符串字段
    if not ans:
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                ans = value.strip()
                logger.info(f"Using fallback answer key '{key}': {ans[:50]}...")
                break
    
    idxs = data.get("support_idxs", [])
    
    # 非空校验
    if not ans:
        raise ValueError("Empty answer in JSON")
    
    # 禁词校验（保持生成为主，但避免明显违规）
    if ans.lower() in FORBIDDEN:
        raise ValueError("Forbidden answer phrase")
    
    # support 至少 1 条（如果确实能在某段落中找到子串）
    # 仅做"存在性"检查，不修改 ans，不做抽取覆盖
    has_substr_somewhere = any(ans in (passages.get(i, "")) for i in idxs)
    if not has_substr_somewhere:
        # 如果模型没给出含子串的 idx，尝试从全部段落里找一条加入（不改答案）
        for i, txt in passages.items():
            if ans and ans in txt:
                idxs = [i]  # 只保底 1 条
                has_substr_somewhere = True
                break
    
    # 去重并转换为int，只保留合法的id（存在于passages_by_idx中的id）
    # support的最终长度由fill_support_idxs_noid统一控制
    seen = set()
    deduplicated_idxs = []
    for i in idxs:
        try:
            idx = int(i)
            # 只保留存在于passages中的合法id
            if idx not in seen and idx in passages:
                seen.add(idx)
                deduplicated_idxs.append(idx)
        except (ValueError, TypeError):
            continue
    idxs = deduplicated_idxs
    
    return ans, idxs

def extract_prediction_with_retry(raw_llm_output: str, passages: Dict[int, str], 
                                retry_func=None, max_retries: int = 1) -> Tuple[str, List[int]]:
    """
    带重试机制的答案提取
    
    Args:
        raw_llm_output: LLM的原始输出
        passages: 段落字典
        retry_func: 重试函数，应该返回新的LLM输出
        max_retries: 最大重试次数
        
    Returns:
        (answer, support_idxs): 答案和支持段落索引列表
    """
    for attempt in range(max_retries + 1):
        try:
            return extract_prediction(raw_llm_output, passages)
        except ValueError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries and retry_func:
                logger.info(f"Retrying with attempt {attempt + 2}")
                raw_llm_output = retry_func()
            else:
                # 最后一次尝试失败，启用兜底机制
                logger.error(f"All attempts failed, activating fallback mechanism")
                logger.debug(f"Raw LLM output causing failure: {raw_llm_output[:500]}")
                
                # 尝试解析JSON结构，检查是否完全为空
                try:
                    parsed_data = parse_llm_json(raw_llm_output)
                    logger.debug(f"Successfully parsed JSON structure: {parsed_data}")
                    
                    # 检查所有可能的答案字段是否都为空
                    answer_keys = ["answer", "final_answer", "prediction"]
                    has_any_answer = False
                    found_answers = {}
                    
                    for key in answer_keys:
                        if key in parsed_data:
                            value = str(parsed_data[key]).strip()
                            found_answers[key] = value
                            if value:
                                has_any_answer = True
                                logger.debug(f"Found non-empty answer in key '{key}': {value[:100]}")
                                break
                    
                    # 如果预定义键都为空，检查是否有任何非空字符串字段
                    if not has_any_answer:
                        logger.debug(f"All predefined answer keys are empty: {found_answers}")
                        for key, value in parsed_data.items():
                            if isinstance(value, str) and value.strip():
                                has_any_answer = True
                                logger.debug(f"Found fallback answer in key '{key}': {value[:100]}")
                                break
                    
                    # 如果完全没有有效答案，返回空答案并记录结构化日志
                    if not has_any_answer:
                        logger.error("Parser fallback activated: no valid answer found in any field", 
                                   extra={"parser_fallback_used": True, 
                                         "raw_output_preview": raw_llm_output[:200],
                                         "parsed_structure": str(parsed_data)[:200],
                                         "all_keys": list(parsed_data.keys()),
                                         "found_answers": found_answers})
                        return "", []
                    
                except Exception as parse_error:
                    # JSON解析完全失败，记录结构化日志并返回空答案
                    logger.error("Parser fallback activated: JSON parsing failed completely", 
                               extra={"parser_fallback_used": True, 
                                     "parse_error": str(parse_error),
                                     "raw_output_preview": raw_llm_output[:200]})
                    return "", []
                
                # 如果能解析但提取失败，使用改进的回退逻辑
                from config import config
                json_parsing_config = config.get('retrieval.json_parsing', {})
                default_fallback = json_parsing_config.get('fallback_message', "Unable to extract a meaningful answer from the provided context")
                
                fallback_answer = raw_llm_output.strip() if raw_llm_output.strip() else "Unable to generate a valid answer"
                
                # 如果原始输出包含空的JSON答案，使用配置的回退消息
                try:
                    parsed_data = parse_llm_json(raw_llm_output)
                    answer_keys = ["answer", "final_answer", "prediction"]
                    all_empty = all(not str(parsed_data.get(key, "")).strip() for key in answer_keys)
                    if all_empty:
                        fallback_answer = default_fallback
                except:
                    pass
                
                fallback_idxs = list(passages.keys())[:3] if passages else []
                logger.info("Using fallback answer after all extraction attempts failed", 
                          extra={"fallback_answer_preview": fallback_answer[:100]})
                return fallback_answer, fallback_idxs
    
    # 这行代码理论上不会执行到
    return "No answer found", []