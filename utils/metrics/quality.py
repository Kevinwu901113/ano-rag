from typing import Dict, Any, List, Optional
from baselines.common.model_clients import get_default_llm_client
from loguru import logger
import re

# --- PROMPTS ---

PROMPT_CONTEXT_RELEVANCE = """
You are an expert evaluator for Retrieval-Augmented Generation systems.
Your task is to evaluate the relevance of the retrieved context to the user's question.

Question: {question}
Retrieved Context:
{context}

Score the relevance on a scale of 1 to 5:
1: Completely irrelevant.
2: Mostly irrelevant, mentions some keywords but misses the point.
3: Partially relevant, contains some useful info but also noise.
4: Mostly relevant, contains the answer but maybe some noise.
5: Highly relevant, contains the exact answer and supporting details cleanly.

Output only the number (1-5).
"""

PROMPT_ANSWER_RELEVANCE = """
You are an expert evaluator.
Your task is to evaluate if the answer actually answers the user's question.

Question: {question}
Answer: {answer}

Score the relevance on a scale of 1 to 5:
1: Completely irrelevant or refuses to answer without reason.
2: Addresses the topic but not the specific question.
3: Partially answers the question.
4: Mostly answers the question directly.
5: Completely answers the question directly and precisely.

Output only the number (1-5).
"""

PROMPT_GROUNDEDNESS = """
You are an expert evaluator.
Your task is to evaluate if the answer is grounded in (supported by) the provided context.
If the answer contains information NOT present in the context (hallucination), penalize it.

Retrieved Context:
{context}

Answer: {answer}

Score the groundedness on a scale of 1 to 5:
1: Not supported at all (hallucination).
2: Mostly unsupported, huge hallucinations.
3: Partially supported, but contains significant external info or minor hallucinations.
4: Mostly supported, minor details might be inferred.
5: Fully supported by the context.

Output only the number (1-5).
"""

class RAGQualityEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.client = get_default_llm_client(config, llm_profile="eval")
        
    def _call_llm(self, prompt: str) -> int:
        try:
            # Enforce temperature=0 for reproducibility
            response = self.client.chat(prompt, history=[], temperature=0.0)
            # Extract number
            match = re.search(r"\b([1-5])\b", response)
            if match:
                return int(match.group(1))
            return 0 # Error
        except Exception as e:
            logger.error(f"LLM Eval failed: {e}")
            return 0

    def evaluate(self, question: str, answer: str, context: str) -> Dict[str, int]:
        """
        Evaluate RAG quality on 3 dimensions.
        """
        results = {}
        
        # Context Relevance
        p_cr = PROMPT_CONTEXT_RELEVANCE.format(question=question, context=context)
        results["context_relevance"] = self._call_llm(p_cr)
        
        # Answer Relevance
        p_ar = PROMPT_ANSWER_RELEVANCE.format(question=question, answer=answer)
        results["answer_relevance"] = self._call_llm(p_ar)
        
        # Groundedness
        p_gr = PROMPT_GROUNDEDNESS.format(answer=answer, context=context)
        results["groundedness"] = self._call_llm(p_gr)
        
        return results

