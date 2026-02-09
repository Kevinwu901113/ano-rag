#!/usr/bin/env python3
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relrag.retriever.llm_parser import LLMQueryParser
from relrag.utils.llm_client import LLMChatClient
from relrag.config.config_loader import config as global_config
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES

# -----------------------------------------------------------------------------
# Discovery Prompt
# -----------------------------------------------------------------------------

DISCOVERY_PROMPT = """
You are an Ontology Engineer expanding the predicate coverage for a Knowledge Graph.
The following question could not be mapped to existing predicates.

Existing Predicates: {existing_preds}

Question: "{question}"

Your task is to propose a NEW canonical predicate that captures the relationship in this question.
If the question is asking about a relationship already covered by existing predicates (or a synonym), please indicate that instead.
But the goal is primarily to discover MISSING predicates (especially for complex relations in HotpotQA/MuSiQue).

Output a JSON object with the following fields:
- "proposed_pred": (string) The canonical name for the new predicate (snake_case, e.g., "drafted_by", "original_language").
- "definition": (string) A one-sentence definition of the predicate.
- "subj_type": (string) The expected type of the subject (e.g., "PERSON", "WORK", "ORG").
- "obj_type": (string) The expected type of the object (e.g., "DATE", "PLACE", "LANGUAGE").
- "surface_forms": (list of strings) Synonyms or trigger phrases (e.g., ["written in", "language of"]).
- "examples": (list of strings) 1-2 example questions that would use this predicate.
- "is_new": (boolean) True if this is a new predicate, False if it maps to an existing one.

Output STRICT JSON only.
"""

def _parse_json_response(content: str) -> Dict[str, Any]:
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        return json.loads(content)
    except Exception:
        return {}

class PredicateDiscoverer:
    def __init__(self):
        self.parser = LLMQueryParser()
        # Create a separate client for discovery to allow different config if needed
        # Reusing the same logic as LLMQueryParser._get_llm_client for consistency
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
            
        self.client = LLMChatClient(
            llm_profile="generate",
            temperature=0.1,
            max_tokens=512,
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key or "sk-no-key-required"
        )
        self.existing_preds_str = ", ".join(sorted(ALLOWED_PREDICATES))

    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = sample.get("question", "")
        if not question:
            return None
        
        # 1. Parse Question
        try:
            ir = self.parser.parse(question)
        except Exception as e:
            logger.error(f"Parser failed for {sample.get('_id')}: {e}")
            return None

        # 2. Check if pred_chain is empty
        if not ir or not ir.pred_chain:
            # Found a candidate!
            # Run Discovery
            return self.run_discovery(sample, question)
        
        return None

    def run_discovery(self, sample: Dict[str, Any], question: str) -> Dict[str, Any]:
        prompt = DISCOVERY_PROMPT.format(
            existing_preds=self.existing_preds_str,
            question=question
        )
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat(messages)
            result = _parse_json_response(response.content)
            if result:
                result["source_id"] = sample.get("_id")
                result["source_question"] = question
            return result
        except Exception as e:
            logger.error(f"Discovery failed for {sample.get('_id')}: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description="Discover missing predicates from dataset")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input dataset JSON (e.g., hotpot_dev_distractor_v1.json)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads")
    
    args = parser.parse_args()
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    data = []
    if input_path.suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to load JSON from {input_path}")
                return
        
    logger.info(f"Loaded {len(data)} samples from {input_path}")
    
    if args.limit > 0:
        data = data[:args.limit]
        logger.info(f"Limiting to first {args.limit} samples")
        
    discoverer = PredicateDiscoverer()
    results = []
    
    # Run processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(discoverer.process_sample, sample) for sample in data]
        
        with open(args.output, "w", encoding="utf-8") as f_out:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                res = future.result()
                if res:
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()
                    results.append(res)
                    
    logger.info(f"Discovery complete. Found {len(results)} new predicate candidates.")
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
