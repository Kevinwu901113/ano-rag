"""鲁棒的JSON解析器，用于处理LLM输出的各种格式问题"""

import json
import re
from typing import Dict, List, Tuple, Any
from loguru import logger

# 禁用的答案短语
FORBIDDEN = {"insufficient information", "no spouse mentioned"}

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
    
    # 3) 处理"字符串化JSON"的情况
    if isinstance(obj, str):
        obj2 = try_load(obj)
        if isinstance(obj2, dict):
            obj = obj2
    
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