import json
import os
import sys
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any, Tuple

from loguru import logger
from relrag.utils.llm_client import LLMChatClient
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES
from relrag.config.config_loader import config as global_config

# -----------------------------------------------------------------------------
# Constants & Prompts
# -----------------------------------------------------------------------------

PREDICATES_JSON_PATH = Path(__file__).resolve().parents[1] / "schema" / "predicates.json"

CLUSTERING_PROMPT_TEMPLATE = """
You are an expert ontology engineer. Your task is to normalize a list of "raw predicates" extracted from text into a set of "Canonical Predicates".

Existing Canonical Predicates:
{existing_canonicals}

Raw Predicates to Cluster (with frequency and examples):
{raw_predicates_block}

Instructions:
1. Group raw predicates that are synonymous or highly related.
2. Map each group to a Canonical Predicate. 
   - PREFER using an Existing Canonical Predicate if it fits.
   - ONLY introduce a NEW Canonical Predicate if the concept is distinct and frequent, and cannot be mapped to existing ones.
3. Ignore very rare or noisy predicates (garbage text).

Output strictly in JSON format:
{{
    "clusters": [
        {{
            "canonical": "existing_or_new_predicate_name",
            "is_new": true/false,
            "aliases": ["raw_pred_1", "raw_pred_2", ...]
        }},
        ...
    ]
}}
"""

# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------

class PredicateLearner:
    def __init__(self, predicates_path: Path = PREDICATES_JSON_PATH):
        self.predicates_path = predicates_path

    def load_notes(self, files: List[Path]) -> List[Dict[str, Any]]:
        notes = []
        for p in files:
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            notes.extend(data)
                        elif isinstance(data, dict):
                            notes.append(data)
                    except json.JSONDecodeError:
                        pass
        return notes

    def extract_raw_predicates(self, notes: List[Dict[str, Any]]) -> Tuple[Counter, Dict[str, List[str]]]:
        counts = Counter()
        examples = defaultdict(list)
        
        for note in notes:
            meta = note.get("meta", {})
            raw = meta.get("raw_pred") or note.get("pred") or ""
            raw = str(raw).strip().lower()
            
            if not raw:
                continue
                
            counts[raw] += 1
            
            evidence = note.get("evidence") or ""
            if evidence and len(examples[raw]) < 3:
                examples[raw].append(evidence)
                
        return counts, examples

    def cluster_predicates(
        self,
        counts: Counter, 
        examples: Dict[str, List[str]], 
        existing_canonicals: Set[str],
        min_freq: int = 5,
        top_k: int = 100
    ) -> Dict[str, List[str]]:
        
        candidates = [p for p, c in counts.most_common(top_k) if c >= min_freq]
        candidates = [p for p in candidates if p not in existing_canonicals]
        
        if not candidates:
            logger.info("No new candidates to cluster (all covered or below freq threshold).")
            return {}

        raw_block_lines = []
        for p in candidates:
            ex_str = " | ".join(examples[p][:2])
            raw_block_lines.append(f"- {p} (freq={counts[p]}): {ex_str}")
        
        raw_block = "\n".join(raw_block_lines)
        existing_str = ", ".join(sorted(list(existing_canonicals)))
        
        prompt = CLUSTERING_PROMPT_TEMPLATE.format(
            existing_canonicals=existing_str,
            raw_predicates_block=raw_block
        )
        
        logger.info(f"Sending {len(candidates)} predicates to LLM for clustering...")
        
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
            api_key = openai_cfg.get("api_key") or os.environ.get(openai_cfg.get("api_key_env", "OPENAI_API_KEY"))

        client = LLMChatClient(
            llm_profile="generate", 
            temperature=0.0, 
            max_tokens=2048,
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key or "sk-no-key-required"
        )
        messages = [{"role": "user", "content": prompt}]
        
        try:
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
            clusters = data.get("clusters", [])
            
            mapping = defaultdict(set)
            for cluster in clusters:
                canon = cluster.get("canonical")
                aliases = cluster.get("aliases", [])
                if not canon:
                    continue
                canon = canon.strip().lower()
                for a in aliases:
                    mapping[canon].add(a.strip().lower())
                    
            final_mapping = {k: list(v) for k, v in mapping.items()}
            return final_mapping
            
        except Exception as e:
            logger.error(f"LLM Clustering failed: {e}")
            return {}

    def merge_mappings(
        self,
        current_map: Dict[str, List[str]], 
        new_map: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        merged = current_map.copy()
        for canon, aliases in new_map.items():
            existing = set(merged.get(canon, []))
            existing.update(aliases)
            merged[canon] = sorted(list(existing))
        return merged

    def run_validator_replay(self, notes_path: Path) -> Dict[str, float]:
        """
        Run a separate process to validate notes using the NEW predicates.json.
        """
        script = f"""
import sys
import json
from pathlib import Path
sys.path.insert(0, "{str(Path(__file__).resolve().parents[2])}")

from relrag.validators.note_validator import validate_and_normalize
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES, PRED_SYNONYM_SETS

def main():
    notes_path = "{str(notes_path)}"
    valid_count = 0
    total = 0
    weak = 0
    raw_preds = 0
    
    with open(notes_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    batch = [obj]
                else:
                    batch = obj
                    
                res = validate_and_normalize(json.dumps(batch), "replay", "0")
                stats = res.get("stats", {{}})
                valid_count += stats.get("valid_count", 0)
                weak += stats.get("weak_predicate_count", 0)
                dropped = stats.get("dropped_predicates", {{}})
                raw_preds += sum(dropped.values())
                
                total += len(batch)
            except Exception:
                pass
                
    metrics = {{
        "valid_rate": valid_count / total if total else 0,
        "weak_rate": weak / total if total else 0,
        "raw_rate": raw_preds / total if total else 0
    }}
    print(json.dumps(metrics))

if __name__ == "__main__":
    main()
"""
        try:
            temp_script = Path("temp_replay.py")
            temp_script.write_text(script, encoding="utf-8")
            
            result = subprocess.run(
                [sys.executable, "temp_replay.py"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            output = result.stdout.strip()
            metrics = json.loads(output)
            
            if temp_script.exists():
                temp_script.unlink()
            return metrics
        except Exception as e:
            logger.error(f"Validation replay failed: {e}")
            if 'result' in locals() and result.stderr:
                logger.error(f"Stderr: {result.stderr}")
            if Path("temp_replay.py").exists():
                Path("temp_replay.py").unlink()
            return {"error": 1.0}

    def learn_and_update(self, input_notes_path: Path, min_freq: int = 5) -> bool:
        """
        Main entry point: Learn from notes, update predicates.json if improved.
        Returns True if updated, False otherwise.
        """
        if not input_notes_path.exists():
            logger.warning(f"Input notes not found: {input_notes_path}")
            return False

        notes = self.load_notes([input_notes_path])
        logger.info(f"Loaded {len(notes)} notes for predicate learning.")
        
        counts, examples = self.extract_raw_predicates(notes)
        logger.info(f"Extracted {len(counts)} unique raw predicates.")
        
        current_mapping = {}
        if self.predicates_path.exists():
            with open(self.predicates_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    current_mapping = data
                elif isinstance(data, list):
                    current_mapping = {k: [] for k in data}
        
        existing_canonicals = set(ALLOWED_PREDICATES)
        
        new_mapping = self.cluster_predicates(counts, examples, existing_canonicals, min_freq=min_freq)
        if not new_mapping:
            logger.info("No new clusters found.")
            return False

        logger.info(f"Proposed new clusters: {json.dumps(new_mapping, indent=2)}")
        proposed_full_mapping = self.merge_mappings(current_mapping, new_mapping)
        
        logger.info("Running baseline validation...")
        base_metrics = self.run_validator_replay(input_notes_path)
        logger.info(f"Baseline: {base_metrics}")
        
        backup_path = self.predicates_path.with_suffix(".json.bak")
        if self.predicates_path.exists():
            self.predicates_path.rename(backup_path)
        
        try:
            with open(self.predicates_path, "w", encoding="utf-8") as f:
                json.dump(proposed_full_mapping, f, indent=2, ensure_ascii=False)
                
            logger.info("Running validation on new predicates...")
            new_metrics = self.run_validator_replay(input_notes_path)
            logger.info(f"New Metrics: {new_metrics}")
            
            improvement = base_metrics.get("weak_rate", 1.0) - new_metrics.get("weak_rate", 1.0)
            logger.info(f"Weak Rate Reduction: {improvement:.4f}")
            
            if improvement > 0:
                logger.info("SUCCESS: Metrics improved. Keeping new predicates.json.")
                if backup_path.exists():
                    backup_path.unlink()
                return True
            else:
                logger.warning("FAILURE: No improvement. Reverting.")
                if backup_path.exists():
                    backup_path.replace(self.predicates_path)
                return False
                
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            if backup_path.exists():
                backup_path.replace(self.predicates_path)
            return False
