"""
Prompt templates for StructRAG baseline (router/structurizer/utilizer).
"""

# Router prompt: pick "chunk" vs "graph"
ROUTER_PROMPT = """You are routing questions to a retrieval pipeline.
Given a question and brief hints about the top candidate documents, choose the best structure type:
- "chunk": when answers can likely be extracted directly from passages.
- "graph": when reasoning over entities/relations is needed or multiple facts must be connected.

Return exactly one word: "chunk" or "graph".

Question:
{question}

Candidate documents:
{documents}

Your answer (one token, lowercase):"""


# Triple extraction prompt
TRIPLE_EXTRACTION_PROMPT = """You are extracting factual triples from a document.
Read the document text and list key triples in JSON array form:
[{{"head": "...", "relation": "...", "tail": "..."}}]
- Keep arguments concise.
- Use nouns for head/tail; relation should be a short verb or predicate.
- If unsure, return an empty list [].

Document title: {title}
Document text:
{content}

Triples JSON:"""


# Sub-question decomposition
DECOMPOSITION_PROMPT = """Break the question into 2-4 focused sub-questions that, when answered, solve the original query.
Return ONLY a JSON array of strings. If no decomposition is needed, return ["{question}"].

Question: {question}
Sub-questions JSON:"""


# Evidence selection
SELECTION_PROMPT = """You will select relevant {item_type}s for each sub-question.
Use the numbered list of {item_type}s below. For each sub-question, pick the most relevant ids (up to 3).
Return a JSON array of objects: [{{"subquestion": "...", "evidence_ids": [id1, id2]}}].

Sub-questions:
{subquestions}

Candidates:
{candidates}

Selections JSON:"""


# Final answer synthesis
FINAL_ANSWER_PROMPT = """You are answering the original question using structured evidence.
Use ONLY the provided sub-questions and their evidence. If evidence is insufficient, reply with "Insufficient evidence".

Original question: {question}

Sub-questions with evidence:
{evidence_block}

Answer:"""
