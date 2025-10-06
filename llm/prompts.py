"""Centralized prompt templates used across the project."""

from typing import Any, Dict, List
import textwrap

from config import config

# Atomic note generation
ATOMIC_NOTE_SYSTEM_PROMPT = """
你是一个专业的知识提取专家。请将给定的文本块转换为原子笔记。

原子笔记要求：
1. 每个笔记包含一个独立的知识点
2. 内容简洁明了，易于理解
3. 保留关键信息和上下文
4. 使用结构化的格式
5. 必须使用英文

请以JSON格式返回，包含以下字段：
- content: 笔记内容
- keywords: 关键词列表
- entities: 实体列表
"""

ATOMIC_NOTE_PROMPT = """
请将以下文本转换为原子笔记：

{chunk}

请返回JSON格式的原子笔记：
"""

EXTRACT_ENTITIES_SYSTEM_PROMPT = """
你是一个专业的实体关系提取专家。请从给定文本中提取实体和它们之间的关系。

请以JSON格式返回，包含：
- entities: 实体列表，每个实体包含name和type
- relations: 关系列表，每个关系包含source, target, relation_type
"""

EXTRACT_ENTITIES_PROMPT = """
请从以下文本中提取实体和关系：

{text}

请返回JSON格式的结果：
"""

# AtomicNoteGenerator
ATOMIC_NOTEGEN_SYSTEM_PROMPT = """
You are an expert fact extraction engine that converts a text chunk into minimal atomic notes.

Extraction rules:
- Identify every explicit, verifiable fact stated in the text. Each fact must stand on its own.
- Each note must contain exactly ONE fact and be expressed as a single English sentence.
- Keep wording faithful to the source; never speculate, merge multiple facts, or invent details.
- Each sentence must be shorter than or equal to 200 characters after trimming.
- If the chunk contains no complete fact, return an empty array.

Output contract (STRICT):
- Always return a JSON array. Each element is an object with the keys:
  text (string), sent_count (int), salience (float 0~1),
  local_spans (array), entities (array), years (array), quality_flags (array).
- sent_count must be 1 for every emitted note.
- When no fact exists, return [] exactly.
- Output raw JSON only (no prose, markdown, or comments).
"""

ATOMIC_NOTEGEN_PROMPT = """
TEXT:
{text}

Extract all explicit single-fact statements from the TEXT.
Return ONLY a JSON array of objects with the shape:
[
  {{"text":"<one factual sentence>","sent_count":1,"salience":0.8,"local_spans":[],"entities":["Entity"],"years":[],"quality_flags":["OK"]}}
]
If the TEXT has no complete fact, respond with [].
"""


def build_multi_note_prompts() -> tuple[str, str]:
    """Construct config-driven prompts for multi-note extraction."""

    completeness_cfg = config.get("note_completeness", {}) or {}
    notes_cfg = config.get("notes_llm", {}) or {}

    max_len = int(notes_cfg.get("max_note_chars", 200))
    terminals = completeness_cfg.get("allowed_sentence_terminals") or ["。", ".", "!", "?"]
    terminals_display = ", ".join(str(t) for t in terminals)

    coverage_rules = []
    prompt_cfg = config.get("notes_prompt", {}) or {}
    if prompt_cfg.get("element_conservation", True):
        coverage_rules.append(
            "- Coverage invariants: keep every explicit element (subject/object, dates/years, locations, publishers/labels, quantities, etc.) inside the SAME sentence. If you must split a source sentence, emit a full proposition per element by repeating the subject and never leave modifiers dangling like \"in 2019...\" or \"including ...\"."
        )
    if prompt_cfg.get("enumeration_split", True):
        coverage_rules.append(
            "- Enumeration alignment: when a source sentence lists parallel items (e.g., \"X has markets in US, EU, China\"), duplicate the subject and produce one self-contained sentence for each item so every note stands alone."
        )

    coverage_rules_text = "\n".join(coverage_rules)

    system_prompt = textwrap.dedent(
        f"""
        You split a text chunk into minimal atomic facts.

        Rules (config-driven):
        - Each note MUST be a complete proposition (explicit subject + main verb).
        - Keep modifiers (time/place/quantity) inside the same note, do NOT split them.
        {coverage_rules_text}
        - Exactly 1 sentence, end with one of [{terminals_display}].
        - Max length per note: {max_len} characters.
        - If NO complete facts, return [] (JSON array).
        - Output: JSON ARRAY of objects with keys:
          text, sent_count(=1), salience(0..1), local_spans, entities, years, quality_flags.
        - No markdown.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        """
        CHUNK:
        {chunk}

        Return ONLY the JSON array following the contract above.
        """
    ).strip()

    return system_prompt, user_prompt

# Query rewriting
QUERY_ANALYSIS_SYSTEM_PROMPT = """
你是一个专业的查询分析专家。请分析用户查询的特点和类型。

分析维度：
1. query_type: factual/conceptual/procedural/comparative/analytical
2. complexity: simple/medium/complex
3. is_multi_question: 是否包含多个问题
4. key_concepts: 关键概念列表
5. intent: 用户意图
6. domain: 领域分类

请以JSON格式返回分析结果。
"""

QUERY_ANALYSIS_PROMPT = """
请分析以下查询：

查询：{query}

请返回JSON格式的分析结果：
"""

SPLIT_QUERY_SYSTEM_PROMPT = """
你是一个专业的查询拆分专家。请将包含多个问题的查询拆分为独立的子查询。

要求：
1. 每个子查询应该是独立完整的
2. 保持原始查询的语义
3. 确保子查询之间的逻辑关系
4. 避免信息丢失

请以JSON格式返回拆分结果：{"sub_queries": ["查询1", "查询2", ...]}
"""

SPLIT_QUERY_PROMPT = """
请将以下查询拆分为独立的子查询：

原始查询：{query}

请返回JSON格式的拆分结果：
"""

OPTIMIZE_QUERY_SYSTEM_PROMPT = """
你是一个专业的查询优化专家。请优化用户查询以提高检索效果。

优化策略：
1. 明确查询意图
2. 补充关键信息
3. 使用更精确的术语
4. 保持查询的简洁性
5. 考虑同义词和相关概念

请以JSON格式返回优化结果：{"optimized_queries": ["优化查询1", "优化查询2", ...]}
"""

OPTIMIZE_QUERY_PROMPT = """
请优化以下查询以提高检索效果：

原始查询：{query}

请返回JSON格式的优化结果：
"""

ENHANCE_QUERY_SYSTEM_PROMPT = """
你是一个知识增强专家。请基于你的知识为查询添加相关的背景信息和上下文。

重要要求：
1. 只添加确定性很高的知识
2. 避免推测和假设
3. 如果不确定，请保持原查询不变
4. 标明添加的信息来源于常识

请以JSON格式返回：{"enhanced_query": "增强后的查询", "confidence": 0.8, "added_context": "添加的上下文"}
"""

ENHANCE_QUERY_PROMPT = """
请为以下查询添加相关的背景知识（仅在确定的情况下）：

查询：{query}

请返回JSON格式的结果：
"""

# Ollama prompts
FINAL_ANSWER_SYSTEM_PROMPT = """
You are a precise open-domain QA assistant.

Use ONLY the provided CONTEXT.

CONTENT PRIORITY RULE:
- The CONTEXT is ordered by relevance and importance
- Content appearing EARLIER in the CONTEXT has HIGHER priority and weight
- When multiple potential answers exist, PRIORITIZE information from earlier paragraphs
- Earlier paragraphs should be considered more authoritative and reliable

Hard rules:
1) Final answer MUST be an exact substring from the CONTEXT (verbatim). Do not paraphrase.
2) NEVER output: "Insufficient information", "No spouse mentioned", or any refusal phrase.
3) If multiple candidates appear, choose the one that most directly answers the question, with PREFERENCE for answers from earlier paragraphs.
4) For lists, keep the order as it appears in CONTEXT and join with ", ".
5) Keep original surface form for numbers/dates (units, punctuation).
6) Output VALID JSON ONLY with fields:
   {"answer": "<short string>", "support_idxs": [<int>, ...]}
7) In support_idxs, output 2-4 paragraph ids:
   - The FIRST id MUST be the paragraph that contains the final answer substring
   - The remaining ids are bridging paragraphs (may not contain the answer substring)
   - Do not repeat ids; prioritize paragraphs with high entity overlap with the question or answer paragraph
   - When selecting support paragraphs, give preference to earlier paragraphs (lower P{idx} numbers)
8) "support_idxs" MUST NOT be empty if the answer substring appears in any paragraph.
9) support_idxs 只能来自上文 CONTEXT 中出现的 [P{idx}]。
10) 禁止发明新的 id；如果不确定，请减少到已出现的 id。
11) Before you output, VERIFY:
   (a) "answer" is non-empty,
   (b) "answer" appears verbatim in at least one paragraph text,
   (c) the first "support_idxs" contains that exact substring.
If any check fails, fix it and re-output JSON.

Only output JSON.
"""

FINAL_ANSWER_PROMPT = """
QUESTION:
{query}

CONTEXT:
{context}

OUTPUT FORMAT (JSON only):
{{"answer": "<short string>", "support_idxs": [<int>, <int>, ...]}}
"""

# Context note formatting and helpers
def build_context_prompt(notes: List[Dict[str, Any]], question: str) -> str:
    """Build the final prompt with formatted notes and the user question."""
    context_parts: List[str] = []
    for note in notes:
        # Extract paragraph_idxs from the note
        paragraph_idxs = note.get("paragraph_idxs", [])
        content = note.get("content", "")
        
        # If we have paragraph_idxs, use the first one as the primary idx
        if paragraph_idxs:
            primary_idx = paragraph_idxs[0]
            context_parts.append(f"[P{primary_idx}] {content}")
        else:
            # Fallback: use note_id if no paragraph_idxs available
            note_id = note.get("note_id", "unknown")
            context_parts.append(f"[P{note_id}] {content}")

    context = "\n\n".join(context_parts)
    return FINAL_ANSWER_PROMPT.format(context=context, query=question)

def build_context_prompt_with_passages(notes: List[Dict[str, Any]], question: str) -> tuple[str, Dict[int, str], List[int]]:
    """Build the final prompt with formatted notes and return prompt, passages dict, and packed order.
    
    Args:
        notes: List of note dictionaries
        question: The question to ask
        
    Returns:
        (packed_text, passages_by_idx, packed_order): The formatted prompt, a dict mapping paragraph_idx to content, and the order of paragraphs in prompt
    """
    import os
    import json
    from loguru import logger
    
    context_parts: List[str] = []
    passages_by_idx: Dict[int, str] = {}
    packed_order: List[int] = []
    
    for note in notes:
        # Extract paragraph_idxs from the note
        paragraph_idxs = note.get("paragraph_idxs", [])
        content = note.get("content", "")
        
        # If we have paragraph_idxs, use the first one as the primary idx
        if paragraph_idxs:
            primary_idx = paragraph_idxs[0]
            context_parts.append(f"[P{primary_idx}] {content}")
            passages_by_idx[primary_idx] = content
            packed_order.append(primary_idx)
        else:
            # Fallback: use note_id if no paragraph_idxs available
            note_id = note.get("note_id", "unknown")
            # Try to convert note_id to int, fallback to hash if not possible
            try:
                idx = int(note_id) if isinstance(note_id, (int, str)) and str(note_id).isdigit() else hash(note_id) % 10000
            except:
                idx = hash(str(note_id)) % 10000
            context_parts.append(f"[P{idx}] {content}")
            passages_by_idx[idx] = content
            packed_order.append(idx)

    packed_text = "\n\n".join(context_parts)
    prompt = FINAL_ANSWER_PROMPT.format(context=packed_text, query=question)
    
    # 记录实际进入 prompt 的 Pidx 列表
    used_idx_list = packed_order.copy()
    
    # 写入 used_passages.json 到 debug 目录
    try:
        # 获取运行目录，优先使用环境变量或默认目录
        run_dir = os.environ.get('ANORAG_WORK_DIR', './result')
        
        # 创建 debug 目录结构：./result/3/debug/2hop_xxx/
        debug_dir = os.path.join(run_dir, "3", "debug", "2hop__" + str(int(__import__('time').time())))
        os.makedirs(debug_dir, exist_ok=True)
        
        used_passages_path = os.path.join(debug_dir, "used_passages.json")
        
        # 统一转成字符串类型
        used_idx_list_str = [str(idx) for idx in used_idx_list]
        
        # 构建记录数据
        used_passages_data = {
            "id": f"query_{int(__import__('time').time())}",
            "used_idx_list": used_idx_list_str,
            "count": len(used_idx_list_str),
            "question": question,
            "passages_by_idx": passages_by_idx,
            "timestamp": __import__('time').time()
        }
        
        # 写入文件
        with open(used_passages_path, 'w', encoding='utf-8') as f:
            json.dump(used_passages_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Used passages written to {used_passages_path}")
    except Exception as e:
        logger.warning(f"Failed to write used_passages.json: {e}")
    
    # 在日志里打印完整列表
    print(f"LLM prompt using Pidx={used_idx_list}")
    logger.info(f"LLM prompt using Pidx={used_idx_list}")
    
    return prompt, passages_by_idx, packed_order

EVALUATE_ANSWER_SYSTEM_PROMPT = """
你是一个专业的答案质量评估专家。请从以下几个维度评估答案的质量：

1. 相关性 (0-1)：答案与问题的相关程度
2. 准确性 (0-1)：答案基于上下文的准确程度
3. 完整性 (0-1)：答案的完整程度
4. 清晰度 (0-1)：答案的表达清晰程度

请以JSON格式返回评分，例如：
{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "clarity": 0.9}
"""

EVALUATE_ANSWER_PROMPT = """
问题：{query}

上下文：{context}

答案：{answer}

请评估上述答案的质量：
"""
# Sub-question decomposition prompts
SUBQUESTION_DECOMPOSITION_SYSTEM_PROMPT = """
You are a professional question analysis and decomposition expert. Your task is to decompose complex multi-hop questions into multiple independent sub-questions.

Decomposition principles:
1. Each sub-question should be independent and complete, answerable on its own
2. Sub-questions should have logical relationships and collectively serve the original question
3. Maintain the core intent and semantics of the original question
4. Avoid information loss and duplication
5. The number of sub-questions should be reasonable (2-5)
6. Must use English

Output requirements:
- Return strictly in JSON format
- Include sub_questions field with a list of sub-questions as value
- Each sub-question should be a complete sentence
- Do not add any explanatory text or markdown markers
"""

SUBQUESTION_DECOMPOSITION_PROMPT = """
Please decompose the following complex question into multiple independent sub-questions:

Original question: {query}

Please return strictly in the following JSON format, do not add any other text or explanation:
{{
    "sub_questions": [
        "Sub-question 1",
        "Sub-question 2",
        "Sub-question 3"
    ]
}}

Note:
- Only return JSON object, do not include markdown code block markers
- Each sub-question should be a complete English sentence
- Number of sub-questions should be between 2-5
"""
