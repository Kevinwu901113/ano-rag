"""Centralized prompt templates used across the project."""

from typing import Any, Dict, List

# Atomic note generation
ATOMIC_NOTE_SYSTEM_PROMPT = """
你是一个专业的知识提取专家。请将给定的文本块逐句处理，提取每句中的原子事实。

【硬性规则 - 必须严格遵守】：
1. 只输出一个JSON数组（[] 开头、] 结尾），不得有任何额外文字、注释、前后缀
2. 当无事实时，返回空数组 []（不要写"无/0/None"等任何文字）
3. 数组元素的字段固定（sent_id, facts, text, entities, predicate, span, source_id），缺失字段用空值/空数组

处理要求：
1. 逐句分析输入文本，为每句分配序号
2. 提取每句中的独立事实，每个事实应简洁完整（≤35词）
3. 去除冗余信息，合并同义表达
4. 事实表达需自洽，包含必要上下文
5. 必须使用英文

字段定义：
- sent_id: 句序号（必需）
- facts: 事实列表（必需，可为空[]）
- text: 事实文本内容（必需）
- entities: 实体列表（可选，默认[]）
- predicate: 谓词/关系（可选，默认""）
- span: 在chunk中的字符跨度[start,end]（可选，默认[]）
- source_id: 来源标识（可选，默认""）

【正例示范】：
输入："Apple was founded in 1976. The weather is nice today."
输出：[{{"sent_id":1,"facts":[{{"text":"Apple was founded in 1976","entities":["Apple"],"predicate":"founded","span":[0,25],"source_id":""}}]}},{{"sent_id":2,"facts":[]}}]

【空数组示范】：
输入："Hello there."
输出：[]

重要：严格按照上述格式，仅输出JSON数组，禁止任何解释文字。
"""

ATOMIC_NOTE_PROMPT = """
请逐句处理以下文本，提取每句中的原子事实：

{chunk}

【硬性规则】：
1. 只输出JSON数组，格式：[{{...}},{{...}}]
2. 无事实时返回空数组：[]
3. 不得有任何解释文字、注释或markdown标记

【正例】：
[{{"sent_id":1,"facts":[{{"text":"Apple was founded in 1976","entities":["Apple"],"predicate":"founded","span":[0,25],"source_id":""}}]}},{{"sent_id":2,"facts":[]}}]

【空例】：
[]

严格按照上述格式输出JSON数组：
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
你是一个专业的知识提取和整理专家。你的任务是将给定的文本逐句处理，提取每句中的原子事实。

【硬性规则 - 必须严格遵守】：
1. 只输出一个JSON数组（[] 开头、] 结尾），不得有任何额外文字、注释、前后缀
2. 当无事实时，返回空数组 []（不要写"无/0/None"等任何文字）
3. 数组元素的字段固定（sent_id, facts, text, entities, predicate, span, source_id），缺失字段用空值/空数组

处理要求：
1. 逐句分析输入文本，为每句分配序号
2. 提取每句中的独立事实，每个事实应简洁完整（≤35词）
3. 去除冗余信息，合并同义表达
4. 事实表达需自洽，包含必要上下文
5. 必须使用英文

字段定义：
- sent_id: 句序号（必需）
- facts: 事实列表（必需，可为空[]）
- text: 事实文本内容（必需）
- entities: 实体列表（可选，默认[]）
- predicate: 谓词/关系（可选，默认""）
- span: 在chunk中的字符跨度[start,end]（可选，默认[]）
- source_id: 来源标识（可选，默认""）

【正例示范】：
输入："Apple was founded in 1976. The weather is nice today."
输出：[{{"sent_id":1,"facts":[{{"text":"Apple was founded in 1976","entities":["Apple"],"predicate":"founded","span":[0,25],"source_id":""}}]}},{{"sent_id":2,"facts":[]}}]

【空数组示范】：
输入："Hello there."
输出：[]

重要：严格按照上述格式，仅输出JSON数组，禁止任何解释文字。
"""

ATOMIC_NOTEGEN_PROMPT = """
请逐句处理以下文本，提取每句中的原子事实。

文本内容：
{text}

【硬性规则】：
1. 只输出JSON数组，格式：[{{...}},{{...}}]
2. 无事实时返回空数组：[]
3. 不得有任何解释文字、注释或markdown标记

【正例】：
[{{"sent_id":1,"facts":[{{"text":"Apple was founded in 1976","entities":["Apple"],"predicate":"founded","span":[0,25],"source_id":""}}]}},{{"sent_id":2,"facts":[]}}]

【空例】：
[]

严格按照上述格式输出JSON数组：
"""

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

请以JSON格式返回拆分结果：{{"sub_queries": ["查询1", "查询2", ...]}}
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

请以JSON格式返回优化结果：{{"optimized_queries": ["优化查询1", "优化查询2", ...]}}
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

请以JSON格式返回：{{"enhanced_query": "增强后的查询", "confidence": 0.8, "added_context": "添加的上下文"}}
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

Hard rules:
1) Final answer MUST be an exact substring from the CONTEXT (verbatim). Do not paraphrase.
2) NEVER output: "Insufficient information", "No spouse mentioned", or any refusal phrase.
3) If multiple candidates appear, choose the one that most directly answers the question.
4) For lists, keep the order as it appears in CONTEXT and join with ", ".
5) Keep original surface form for numbers/dates (units, punctuation).
6) Output VALID JSON ONLY with fields:
   {{"answer": "<short string>", "support_idxs": [<int>, ...]}}
7) In support_idxs, output 2-4 paragraph ids:
   - The FIRST id MUST be the paragraph that contains the final answer substring
   - The remaining ids are bridging paragraphs (may not contain the answer substring)
   - Do not repeat ids; prioritize paragraphs with high entity overlap with the question or answer paragraph
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
    return prompt, passages_by_idx, packed_order

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


