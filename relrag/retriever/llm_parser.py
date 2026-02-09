import json
import re
from typing import Optional, List, Any, Dict
from loguru import logger

from relrag.utils.llm_client import LLMChatClient
from relrag.config.config_loader import config as global_config
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES
from relrag.retriever.ir import QueryIR, Seed, PredicateStep

# -----------------------------------------------------------------------------
# LLM Parser Prompt
# -----------------------------------------------------------------------------

QUERY_PARSER_PROMPT = """
You are a semantic parser for a Knowledge Graph Retrieval System.
Your task is to parse a natural language question into a structured Query Intermediate Representation (QueryIR).

Target Schema:
{{
    "seeds": [
        {{ "text": "Entity Name", "type_hint": "PERSON/ORG/WORK/PLACE/etc" }}
    ],
    "pred_chain": [
        {{ "pred": "predicate_name", "direction": "out/in", "target_hint": "TargetType" }}
    ],
    "max_hops": 2
}}

Constraints:
1. "seeds": Extract the main entity or entities from the question.
2. "pred_chain": A sequence of predicates to traverse from the seed.
3. "direction": "out" (Seed -> Target) or "in" (Target -> Seed).
4. "max_hops": Usually 1 or 2.

ALLOWED PREDICATES (Strictly enforce this list):
{allowed_predicates}

Instructions:
- If the question is a simple lookup (e.g., "Who directed X?"), chain is ["directed_by"].
- If the question is multi-hop (e.g., "Who is the spouse of the director of X?"), chain is ["directed_by", "spouse"].
- If the predicate is NOT in the allowed list, try to map it to the closest one or output valid sub-parts.
- If NO predicates match, leave "pred_chain" empty (open entity query).
- Output STRICT JSON only.

Question: {question}
"""

class LLMQueryParser:
    def __init__(self):
        pass

    def _get_llm_client(self) -> LLMChatClient:
        # Reusing the same client config logic
        cfg = global_config.load_config()
        openai_cfg = cfg.get("openai") or {}
        vllm_cfg = cfg.get("vllm") or {}
        
        provider = "vllm"
        endpoint = vllm_cfg.get("endpoint")
        model = vllm_cfg.get("model")
        api_key = vllm_cfg.get("api_key")
        
        if openai_cfg.get("enabled"):
            provider = "openai"
            endpoint = openai_cfg.get("base_url")
            model = openai_cfg.get("model")
            api_key = openai_cfg.get("api_key")
            
        return LLMChatClient(
            llm_profile="generate", # Use generate profile for low latency
            temperature=0.0,
            max_tokens=512,
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key or "sk-no-key-required"
        )

    def parse(self, question: str) -> Optional[QueryIR]:
        client = self._get_llm_client()
        
        allowed_str = ", ".join(sorted(ALLOWED_PREDICATES))
        prompt = QUERY_PARSER_PROMPT.format(
            allowed_predicates=allowed_str,
            question=question
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # logger.info(f"LLM Parsing question: {question}")
            response = client.chat(messages)
            content = response.content
            
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0]
            elif "{" in content:
                start = content.find("{")
                end = content.rfind("}")
                json_match = content[start:end+1]
                
            data = json.loads(json_match)
            return self._build_query_ir(data, question)
            
        except Exception as e:
            logger.warning(f"LLM Query Parsing failed: {e}")
            return None

    def _build_query_ir(self, data: Dict[str, Any], raw_question: str) -> Optional[QueryIR]:
        seeds_data = data.get("seeds", [])
        if not seeds_data:
            return None
            
        seeds = [Seed(text=s.get("text", ""), type_hint=s.get("type_hint")) for s in seeds_data if s.get("text")]
        if not seeds:
            return None
            
        chain_data = data.get("pred_chain", [])
        chain = []
        valid_preds = set(ALLOWED_PREDICATES)
        
        for step in chain_data:
            pred = step.get("pred", "").lower()
            if pred in valid_preds:
                chain.append(PredicateStep(
                    pred=pred,
                    direction=step.get("direction", "out"),
                    target_hint=step.get("target_hint")
                ))
            else:
                logger.warning(f"LLM predicted invalid predicate: {pred}. Dropping step.")
                # Optional: We could stop here or continue. Dropping might make the chain broken.
                # If strict, we might return None. Here we drop the step to fallback to partial or entity query.
                
        intent = "relation_query" if chain else "open_entity_query"
        max_hops = data.get("max_hops", 2)
        
        return QueryIR(
            intent=intent,
            seeds=seeds,
            pred_chain=chain,
            max_hops=max(max_hops, len(chain)),
            fanout=12,
            raw=raw_question,
            fallback=not chain # If no chain found, mark as fallback/open query
        )
