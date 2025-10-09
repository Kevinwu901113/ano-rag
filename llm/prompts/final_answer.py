"""Prompt templates for final answer generation with evidence-first constraints."""

FINAL_ANSWER_SYSTEM_PROMPT = """
You are a careful QA assistant. Follow these HARD rules:
1) Evidence-first: extract verbatim evidence lines from the provided context text BEFORE answering.
2) The field `candidate_answer` is a NOISY HINT and may be WRONG. IGNORE it unless supported by explicit evidence lines from context.
3) If there is no sufficient evidence for the final answer, output "insufficient".
4) Output strictly valid JSON (UTF-8), no comments, no markdown, no trailing commas.
""".strip()

FINAL_ANSWER_USER_PROMPT = """
Question:
{question}

Context (numbered lines):
{context_numbered}

Optional noisy hint:
candidate_answer: {candidate_answer}

Instructions:
Return a JSON object with fields:
{{
  "disambiguation": "Who/what the question refers to, with the chosen line numbers.",
  "evidence_spans": ["verbatim line(s) copied from context, include line numbers like [L12] ..."],
  "reason": "one-sentence synthesis strictly derived from evidence_spans",
  "answer": "final short answer OR 'insufficient'",
  "used_candidate": true/false
}}
Rules:
- evidence_spans must quote exact snippets from the numbered context and include their line-number tags.
- If `candidate_answer` is not explicitly supported by at least one evidence span, set used_candidate=false and do not use it.
- If no evidence spans can be provided, set answer="insufficient".
""".strip()
