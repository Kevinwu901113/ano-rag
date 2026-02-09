#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path

# Try to import openai, handle if missing (though expected in this env)
try:
    from openai import AsyncOpenAI, APIError, APITimeoutError
except ImportError:
    print("Error: 'openai' package is required. Please install it.")
    sys.exit(1)

# --- Configuration & Constants ---

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "qwen3-30b-a3b"  # Just a placeholder, user should provide
SYSTEM_PROMPT = (
    "You are an exact scoring engine for QA evaluation. "
    "You must follow the scoring rules exactly and output ONLY valid JSON. "
    "Do not include any explanations."
)

SCORING_RULES = """
**Normalization Rules**:
1. Lowercase the text.
2. Remove punctuation (remove characters that are not alphanumeric or whitespace).
3. Remove articles ("a", "an", "the") as standalone words.
4. Normalize whitespace (replace multiple spaces with single space, trim).

**Metrics**:
1. **Exact Match (EM)**:
   - Apply normalization to both prediction and gold.
   - Score is 1 if normalized_pred equals normalized_gold, else 0.

2. **F1 Score**:
   - Apply normalization to both prediction and gold.
   - Tokenize by splitting on whitespace.
   - Compute F1 based on MULTISET token overlap (bag of words).
   - Precision = common_tokens / len(pred_tokens)
   - Recall = common_tokens / len(gold_tokens)
   - F1 = 2 * Precision * Recall / (Precision + Recall)
   - If pred or gold is empty/zero length, handle gracefully (F1=0 usually, unless both empty).

**Multiple Gold Answers**:
- If multiple gold answers are provided, calculate EM and F1 against EACH gold answer.
- Final EM = max(EM scores)
- Final F1 = max(F1 scores)
- For the "best" gold index, pick the one that gives the max F1 (tie-break with EM, then index).
"""

OUTPUT_SCHEMA = """
{
  "final_em": 0 or 1,
  "final_f1": number,
  "best_f1_gold_index": integer,
  "best_em_gold_index": integer,
  "debug": {
    "normalized_pred": string,
    "per_gold": [
      {
        "normalized_gold": string,
        "em": 0 or 1,
        "f1": number,
        "pred_tokens": [string],
        "gold_tokens": [string],
        "common": integer,
        "precision": number,
        "recall": number
      }
    ]
  }
}
"""

# --- Data Classes ---

@dataclass
class EvalSample:
    id: str
    question: str
    pred: str
    golds: List[str]
    official_em: Optional[float] = None
    official_f1: Optional[float] = None

@dataclass
class JudgeResult:
    final_em: int
    final_f1: float
    best_f1_gold_index: int
    best_em_gold_index: int
    debug: Dict[str, Any]
    error: Optional[str] = None
    raw_output: Optional[str] = None
    retries: int = 0
    cache_hit: bool = False

# --- Utilities ---

def calculate_hash(args: argparse.Namespace, prompt_hash: str, sample: EvalSample) -> str:
    # cache_key = hash(model + prompt_hash + question + pred + golds_json)
    # We also include temperature/top_p implicit in prompt/setup, but user says explicit fields.
    # To be safe, include model name.
    golds_str = json.dumps(sample.golds, sort_keys=True)
    content = f"{args.model}|{prompt_hash}|{sample.question}|{sample.pred}|{golds_str}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def infer_field(row: Dict[str, Any], keys: List[str], mandatory: bool = True) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    if mandatory:
        # Try case-insensitive matching
        row_keys_lower = {k.lower(): k for k in row.keys()}
        for k in keys:
            if k.lower() in row_keys_lower:
                return row[row_keys_lower[k.lower()]]
    return None

def parse_sample(row: Dict[str, Any], args: argparse.Namespace) -> Optional[EvalSample]:
    # ID
    id_keys = [args.id_key] if args.id_key else ["id", "_id", "qid", "question_id"]
    sid = infer_field(row, id_keys)
    if sid is None:
        return None  # Skip rows without ID

    # Question
    q_keys = [args.question_key] if args.question_key else ["question", "query"]
    question = infer_field(row, q_keys)
    if question is None:
        return None

    # Pred
    p_keys = [args.pred_key] if args.pred_key else ["pred", "prediction", "answer", "output", "short_answer", "answer_final"]
    pred = infer_field(row, p_keys)
    if pred is None:
        pred = "" # Allow empty prediction
    if not isinstance(pred, str):
        pred = str(pred)

    # Gold
    g_keys = [args.gold_key] if args.gold_key else ["gold", "answers", "ground_truth", "label", "answer"]
    gold = infer_field(row, g_keys)
    if gold is None:
        # Maybe it's in 'supporting_facts' or something? No, user said 'gold'.
        # Check specific dataset formats like HotpotQA where 'answer' is the gold string.
        # If 'answer' was used for pred, we might have a conflict if both keys exist.
        # But usually input is a prediction file which might carry 'answer' as pred or gold.
        # Let's rely on priority.
        # If row has "prediction" and "answer", likely "prediction" is pred, "answer" is gold.
        pass
    
    # Fallback logic for gold if primary inference failed or collided
    if gold is None:
        # Assuming typical prediction file might just have "answer" as gold if "prediction" is present
        if "prediction" in row and "answer" in row:
            gold = row["answer"]
        elif "pred" in row and "answer" in row:
            gold = row["answer"]

    if gold is None:
        return None # Cannot eval without gold

    golds = []
    if isinstance(gold, str):
        golds = [gold]
    elif isinstance(gold, list):
        golds = [str(g) for g in gold]
    else:
        golds = [str(gold)]

    # Official stats
    off_em = infer_field(row, ["official_em", "em"], mandatory=False)
    off_f1 = infer_field(row, ["official_f1", "f1"], mandatory=False)

    return EvalSample(
        id=str(sid),
        question=str(question),
        pred=pred,
        golds=golds,
        official_em=float(off_em) if off_em is not None else None,
        official_f1=float(off_f1) if off_f1 is not None else None
    )

class LLMJudge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env, DEFAULT_API_KEY)
        )
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.prompt_hash = self._compute_prompt_hash()

    def _compute_prompt_hash(self) -> str:
        # Hash of System + Rules + Output Schema
        content = SYSTEM_PROMPT + SCORING_RULES + OUTPUT_SCHEMA
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _build_user_prompt(self, sample: EvalSample, is_retry: bool = False, prev_output: str = "") -> str:
        if is_retry:
            return (
                f"Your previous output was invalid JSON or did not follow schema.\n"
                f"Previous Output:\n{prev_output}\n\n"
                f"Please fix it and return ONLY the valid JSON complying with the schema below.\n"
                f"{OUTPUT_SCHEMA}"
            )
        
        return (
            f"**Scoring Rules**:\n{SCORING_RULES}\n\n"
            f"**Input**:\n"
            f"Question: {sample.question}\n"
            f"Prediction: {sample.pred}\n"
            f"Gold Answers: {json.dumps(sample.golds)}\n\n"
            f"**Output Schema** (JSON ONLY):\n{OUTPUT_SCHEMA}"
        )

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=self.args.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM Call Failed: {str(e)}")

    def _validate_json(self, content: str) -> Dict[str, Any]:
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content)
        
        # Schema check
        required_keys = {"final_em", "final_f1", "best_f1_gold_index", "debug"}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Missing keys: {required_keys - data.keys()}")
        
        # Value check
        if data["final_em"] not in [0, 1]:
            raise ValueError(f"final_em must be 0 or 1, got {data['final_em']}")
        
        f1 = float(data["final_f1"])
        if not (0.0 <= f1 <= 1.0):
             # Allow tiny float error tolerance if needed, but usually 0-1
             if f1 < -1e-6 or f1 > 1.0 + 1e-6:
                 raise ValueError(f"final_f1 out of range [0,1]: {f1}")
        
        return data

    async def judge(self, sample: EvalSample) -> JudgeResult:
        async with self.semaphore:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(sample)}
            ]
            
            last_error = None
            raw_output = ""
            
            for attempt in range(3): # Initial + 2 Retries
                try:
                    raw_output = await self._call_llm(messages)
                    data = self._validate_json(raw_output)
                    return JudgeResult(
                        final_em=data["final_em"],
                        final_f1=float(data["final_f1"]),
                        best_f1_gold_index=data.get("best_f1_gold_index", 0),
                        best_em_gold_index=data.get("best_em_gold_index", 0),
                        debug=data["debug"],
                        raw_output=raw_output,
                        retries=attempt
                    )
                except Exception as e:
                    last_error = str(e)
                    # Prepare retry prompt
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": self._build_user_prompt(sample, is_retry=True, prev_output=raw_output)}
                    ]
                    # Don't sleep to keep it fast, but maybe yield
                    await asyncio.sleep(0.1)
            
            # Failed
            return JudgeResult(
                final_em=0,
                final_f1=0.0,
                best_f1_gold_index=-1,
                best_em_gold_index=-1,
                debug={},
                error=f"Max retries reached. Last error: {last_error}",
                raw_output=raw_output
            )

# --- Main Logic ---

async def main_async():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for EM/F1 Evaluation")
    
    # I/O
    parser.add_argument("--input", required=True, help="Input predictions.jsonl file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    
    # LLM
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--max_tokens", type=int, default=800)
    
    # Field Mapping
    parser.add_argument("--id_key", help="Key for ID")
    parser.add_argument("--question_key", help="Key for Question")
    parser.add_argument("--pred_key", help="Key for Prediction")
    parser.add_argument("--gold_key", help="Key for Gold Answer")
    
    # Control
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--only_failed_official", action="store_true", help="Only judge if official_em=0 or official_f1 < threshold")
    parser.add_argument("--official_f1_threshold", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Setup
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    judge_file = out_dir / "judge_emf1.jsonl"
    cache_file = out_dir / "cache.jsonl"
    summary_json = out_dir / "summary_llm_judge.json"
    summary_md = out_dir / "summary_llm_judge.md"
    
    # Load Cache
    cache = {}
    if cache_file.exists():
        print(f"Loading cache from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "cache_key" in item:
                        cache[item["cache_key"]] = item
                except:
                    pass
    
    # Load Samples
    print(f"Loading predictions from {input_path}...")
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample = parse_sample(row, args)
            if sample:
                samples.append(sample)
    
    print(f"Found {len(samples)} valid samples.")
    
    # Filtering
    filtered_samples = []
    for s in samples:
        if args.only_failed_official:
            # Check official stats
            # If official stats are missing, we MUST judge it (safe default)
            should_judge = True
            if s.official_em is not None and s.official_f1 is not None:
                if s.official_em == 1 and s.official_f1 >= args.official_f1_threshold:
                    should_judge = False
            if should_judge:
                filtered_samples.append(s)
        else:
            filtered_samples.append(s)
            
    if args.limit:
        filtered_samples = filtered_samples[:args.limit]
        print(f"Limiting to {len(filtered_samples)} samples.")
    else:
        print(f"Processing {len(filtered_samples)} samples after filtering.")

    # Init Judge
    judge_engine = LLMJudge(args)
    
    # Processing
    results = []
    
    # Prepare cache keys and check hits
    tasks = []
    cache_hits = 0
    
    # Open file handles for appending results/cache
    f_judge = open(judge_file, "a", encoding="utf-8")
    f_cache = open(cache_file, "a", encoding="utf-8")
    
    async def process_one(sample):
        key = calculate_hash(args, judge_engine.prompt_hash, sample)
        
        # Check Cache
        if key in cache:
            cached_data = cache[key]
            # Reconstruct result from cache
            return JudgeResult(
                final_em=cached_data["final_em"],
                final_f1=cached_data["final_f1"],
                best_f1_gold_index=cached_data.get("best_f1_gold_index", 0),
                best_em_gold_index=cached_data.get("best_em_gold_index", 0),
                debug=cached_data.get("debug", {}),
                cache_hit=True
            ), key, sample
        
        # Run Judge
        res = await judge_engine.judge(sample)
        return res, key, sample

    # Run batch
    # We use as_completed to write as we go
    pending = [process_one(s) for s in filtered_samples]
    
    # Progress bar
    total = len(pending)
    completed = 0
    
    stats = Counter()
    
    for future in asyncio.as_completed(pending):
        res, key, sample = await future
        completed += 1
        
        if res.cache_hit:
            stats["cache_hits"] += 1
        elif res.error:
            stats["errors"] += 1
            print(f"Error for {sample.id}: {res.error}")
        else:
            stats["success"] += 1
            # Write to cache
            cache_entry = {
                "cache_key": key,
                "id": sample.id,
                "final_em": res.final_em,
                "final_f1": res.final_f1,
                "best_f1_gold_index": res.best_f1_gold_index,
                "best_em_gold_index": res.best_em_gold_index,
                "debug": res.debug,
                "raw_output": res.raw_output
            }
            f_cache.write(json.dumps(cache_entry) + "\n")
            f_cache.flush()
        
        # Write to judge output (always, even if cache hit, so we have a full result file for this run)
        out_entry = {
            "id": sample.id,
            "question": sample.question,
            "pred": sample.pred,
            "golds": sample.golds,
            "llm_judge": {
                "em": res.final_em,
                "f1": res.final_f1,
                "best_gold_idx": res.best_f1_gold_index,
                "debug": res.debug,
                "error": res.error,
                "cache_hit": res.cache_hit
            }
        }
        f_judge.write(json.dumps(out_entry) + "\n")
        f_judge.flush()
        
        # Simple progress
        if completed % 10 == 0 or completed == total:
            print(f"Progress: {completed}/{total} (Hits: {stats['cache_hits']}, Errors: {stats['errors']})", end="\r")
            
    print("\nDone.")
    f_judge.close()
    f_cache.close()
    
    # Calculate Summary
    # Reload judge file to calculate accurate summary of THIS run
    all_ems = []
    all_f1s = []
    bad_count = 0
    unstable_count = 0 # Not tracked currently (requires repeated sampling), we assume retries handled it
    
    # Re-read the file we just wrote (or appended to)
    # Wait, if we appended, we might have old results? 
    # The user asked for "out_dir/judge_emf1.jsonl". If we append, we mix runs. 
    # But usually eval is per run.
    # We should probably clear judge_emf1.jsonl at start if not resuming?
    # User said "断点续跑 (resume)". 
    # If resume, we should skip ALREADY PROCESSED IDs in judge_emf1.jsonl.
    # My current logic re-processes but uses cache. That writes duplicate lines to judge_emf1.jsonl.
    # I should improve this: check if ID exists in judge_emf1.jsonl and skip adding to `pending`.
    
    # (Refinement for Resume)
    # Ideally, we read judge_emf1.jsonl first.
    # But for now, let's just calculate summary from the results list we could have collected.
    # I didn't collect `results` in memory to save RAM, but I can re-read the file.
    
    # Let's re-read the judge output for summary.
    print("Generating summary...")
    final_ids = set()
    with open(judge_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row["id"] in final_ids: continue # naive dedupe if multiple runs
                final_ids.add(row["id"])
                
                j = row.get("llm_judge", {})
                if j.get("error"):
                    bad_count += 1
                else:
                    all_ems.append(j.get("em", 0))
                    all_f1s.append(j.get("f1", 0.0))
            except:
                pass

    avg_em = sum(all_ems) / len(all_ems) if all_ems else 0.0
    avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_hash": judge_engine.prompt_hash,
        "samples_count": len(all_ems) + bad_count,
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "bad_count": bad_count,
        "cache_hits": stats["cache_hits"]
    }
    
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    md_content = f"""# LLM Judge Summary

- **Model**: {args.model}
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Samples**: {summary['samples_count']}
- **Avg EM**: {avg_em:.4f}
- **Avg F1**: {avg_f1:.4f}
- **Failed**: {bad_count}
- **Cache Hits**: {stats['cache_hits']}

## Configuration
- Base URL: `{args.base_url}`
- Prompt Hash: `{summary['prompt_hash']}`
"""
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Summary written to {summary_json}")
    print(f"Avg EM: {avg_em:.4f}, Avg F1: {avg_f1:.4f}")

if __name__ == "__main__":
    asyncio.run(main_async())
