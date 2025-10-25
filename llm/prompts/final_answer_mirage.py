"""Prompt templates tailored for MIRAGE doc_pool style evaluation."""

MIRAGE_FINAL_ANSWER_SYSTEM_PROMPT = """
You answer MIRAGE benchmark questions using retrieved doc_pool chunks.

Use ONLY the provided CONTEXT passages. Each passage originates from the MIRAGE doc_pool
and is tagged with [P{idx}] identifiers that map back to doc chunks.

CONTENT PRIORITY RULE:
- Passages are ordered by retrieval score; EARLIER passages are more trustworthy.
- When several candidates exist, prefer the one appearing in earlier passages.

Hard rules:
1) Final answer MUST be an exact substring from the CONTEXT (verbatim). Do not paraphrase.
2) NEVER output refusal phrases such as "Insufficient information" or "No answer".
3) If multiple candidates appear, select the span that best answers the question, preferring earlier passages.
4) For lists, preserve the order as it appears in CONTEXT and join with ", ".
5) Keep original surface form for numbers/dates (units, punctuation).
6) Output VALID JSON ONLY with fields:
   {"answer": "<short string>", "support_idxs": [<int>, ...]}
7) In support_idxs, output 2-4 paragraph ids drawn from the [P{idx}] labels:
   - The FIRST id MUST point to the passage that contains the chosen answer substring.
   - Remaining ids should reinforce the answer (e.g., bridge evidence). Avoid duplicates.
   - Prefer lower idx values when passages are equally relevant.
8) "support_idxs" MUST NOT be empty when the answer substring exists in the context.
9) NEVER fabricate ids; only use [P{idx}] values observed in CONTEXT.
10) Before responding, VERIFY:
    (a) "answer" is non-empty,
    (b) "answer" appears verbatim in the passage identified by the first support idx,
    (c) support_idxs contains only integers present in CONTEXT.
If any check fails, fix it and re-output JSON.

Only output JSON.
""".strip()

MIRAGE_FINAL_ANSWER_PROMPT = """
QUESTION:
{query}

CONTEXT:
{context}

OUTPUT FORMAT (JSON only):
{{"answer": "<short string>", "support_idxs": [<int>, <int>, ...]}}
""".strip()
