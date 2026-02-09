import json
import re
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

from loguru import logger
from relrag.utils.llm_client import LLMChatClient
from relrag.config.config_loader import config as global_config

# -----------------------------------------------------------------------------
# Constants & Prompts
# -----------------------------------------------------------------------------

ATTRIBUTES_YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "attributes.yaml"
VOCAB_JSON_PATH = Path(__file__).resolve().parent / "vocab.json"
ALIASES_JSON_PATH = Path(__file__).resolve().parent / "aliases.json"

ATTRIBUTE_MINING_PROMPT = """
You are an ontology engineer. Analyze the following list of "definition sentences" extracted from a corpus.
Your goal is to define structured attributes for a schema.

Input Sentences (Grouped by apparent attribute):
{patterns_block}

Existing Attributes:
{existing_attributes}

Instructions:
1. For each group, define a canonical attribute name (snake_case).
2. Write a clear "definition" for human understanding.
3. Identify "negative_verbs" (verbs that imply the attribute is FALSE, if any).
4. List "aliases" (phrases that indicate this attribute).
5. Extract a "value_lexicon" (list of common valid values seen in the examples).

Output strictly in YAML format:
attributes:
  attribute_name:
    definition: "..."
    negative_verbs: ["...", "..."]
    aliases: ["...", "..."]
    value_lexicon: ["...", "..."]
"""

VOCAB_REFINEMENT_PROMPT = """
You are a linguist specializing in entity resolution.
Your task is to merge synonymous terms and disambiguate canonical forms for a specific domain slot: "{slot}".

Input Candidates (Term -> Frequency):
{candidates_block}

Instructions:
1. Identify synonymous terms and group them under a single "canonical" form.
2. The canonical form should be the most standard/formal representation.
3. Provide "evidence" (a short reason or example) for the grouping.

Output strictly in JSON format:
{
    "groups": [
        {
            "canonical": "Standard Form",
            "aliases": ["alias1", "alias2"],
            "evidence": "..."
        },
        ...
    ]
}
"""

# -----------------------------------------------------------------------------
# Schema Learner
# -----------------------------------------------------------------------------

class SchemaLearner:
    def __init__(self):
        self.attributes_path = ATTRIBUTES_YAML_PATH
        self.vocab_path = VOCAB_JSON_PATH
        self.aliases_path = ALIASES_JSON_PATH

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

    def _get_llm_client(self) -> LLMChatClient:
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
            llm_profile="generate",
            temperature=0.0,
            max_tokens=4096,
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key or "sk-no-key-required"
        )

    # -------------------------------------------------------------------------
    # Attribute Learning
    # -------------------------------------------------------------------------

    def learn_attributes(self, notes_path: Path, output_path: Optional[Path] = None) -> bool:
        notes = self.load_notes([notes_path])
        logger.info(f"Loaded {len(notes)} notes for attribute learning.")
        
        patterns = self._mine_definition_patterns(notes)
        if not patterns:
            logger.info("No definition patterns found.")
            return False
            
        logger.info(f"Mined {len(patterns)} pattern groups.")
        
        existing_attrs = {}
        if self.attributes_path.exists():
            with open(self.attributes_path, "r", encoding="utf-8") as f:
                existing_attrs = yaml.safe_load(f) or {}

        new_attrs = self._generate_attributes_with_llm(patterns, existing_attrs)
        if not new_attrs:
            return False

        # Validation (Simple Hit Rate Check)
        valid_attrs = self._validate_attributes(new_attrs, notes)
        
        # Merge
        final_attrs = existing_attrs.copy()
        final_attrs.update(valid_attrs)
        
        target_path = output_path or self.attributes_path
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(final_attrs, f, sort_keys=False, allow_unicode=True)
            
        logger.info(f"Saved {len(final_attrs)} attributes to {target_path}")
        return True

    def _mine_definition_patterns(self, notes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract sentences that look like definitions: "X is a Y", "X's occupation is Y".
        Returns: { 'potential_attribute_key': ['example sentence', ...] }
        """
        patterns = defaultdict(list)
        
        # Simple heuristics for definition sentences
        # 1. "is a/an [CONCEPT]"
        # 2. "[PROP] of [ENTITY] is [VALUE]"
        
        for note in notes:
            evidence = note.get("evidence", "")
            if not evidence:
                continue
            
            # Pattern 1: "... is a/an ..." (Capture the concept)
            match = re.search(r"\b(is|was)\s+an?\s+([a-zA-Z\s]+)\b", evidence, re.I)
            if match:
                concept = match.group(2).strip().lower()
                if len(concept.split()) <= 3: # Keep it short
                    patterns[f"is_a_{concept}"].append(evidence)
                    
            # Pattern 2: "... occupation is ..."
            match = re.search(r"\b(occupation|profession|role|title|nationality)\s+(?:of\s+[^is]+\s+)?(?:is|was)\s+([^,.]+)", evidence, re.I)
            if match:
                attr = match.group(1).lower()
                patterns[attr].append(evidence)
                
        # Filter groups with too few examples
        return {k: v[:5] for k, v in patterns.items() if len(v) >= 3}

    def _generate_attributes_with_llm(self, patterns: Dict[str, List[str]], existing: Dict) -> Dict[str, Any]:
        patterns_block = []
        for key, examples in patterns.items():
            ex_str = "\n".join([f"- {e}" for e in examples])
            patterns_block.append(f"Group '{key}':\n{ex_str}\n")
            
        prompt = ATTRIBUTE_MINING_PROMPT.format(
            patterns_block="\n".join(patterns_block),
            existing_attributes=yaml.dump(existing)
        )
        
        client = self._get_llm_client()
        messages = [{"role": "user", "content": prompt}]
        
        try:
            logger.info("Sending attribute patterns to LLM...")
            response = client.chat(messages)
            content = response.content
            
            # Extract YAML
            yaml_match = content
            if "```yaml" in content:
                yaml_match = content.split("```yaml")[1].split("```")[0]
            elif "```" in content:
                yaml_match = content.split("```")[1].split("```")[0]
                
            data = yaml.safe_load(yaml_match)
            return data.get("attributes", {})
        except Exception as e:
            logger.error(f"LLM Attribute Generation failed: {e}")
            return {}

    def _validate_attributes(self, attributes: Dict[str, Any], notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate attributes by checking if their aliases hit anything in the notes.
        """
        valid = {}
        for name, defn in attributes.items():
            aliases = defn.get("aliases", [])
            if not aliases:
                continue
            
            hits = 0
            for note in notes:
                ev = note.get("evidence", "").lower()
                for alias in aliases:
                    if alias.lower() in ev:
                        hits += 1
                        break
            
            hit_rate = hits / len(notes) if notes else 0
            logger.info(f"Attribute '{name}' hit rate: {hit_rate:.4f}")
            
            # Threshold: Just needs to hit *something* to be valid (e.g., > 0.1% or count > 5)
            if hits >= 3:
                valid[name] = defn
                
        return valid

    # -------------------------------------------------------------------------
    # Vocab Learning
    # -------------------------------------------------------------------------

    def learn_vocab(self, notes_path: Path, output_dir: Optional[Path] = None) -> bool:
        notes = self.load_notes([notes_path])
        candidates = self._extract_vocab_candidates(notes)
        
        if not candidates:
            logger.info("No vocab candidates found.")
            return False
            
        logger.info(f"Extracted candidates for slots: {list(candidates.keys())}")
        
        updates = {}
        for slot, counts in candidates.items():
            refined = self._refine_vocab_with_llm(slot, counts)
            if refined:
                updates[slot] = refined

        if not updates:
            return False
            
        # Update files
        self._update_vocab_files(updates, output_dir)
        return True

    def _extract_vocab_candidates(self, notes: List[Dict[str, Any]]) -> Dict[str, Counter]:
        """
        Extract potential vocab terms.
        Focus on: 'nationality', 'occupation', 'title' (mapped from preds)
        """
        candidates = defaultdict(Counter)
        
        for note in notes:
            pred = note.get("pred")
            obj = note.get("obj")
            
            if not pred or not obj:
                continue
                
            # Map pred to slot
            slot = None
            if pred in {"nationality", "citizenship"}:
                slot = "nationality"
            elif pred in {"occupation", "profession", "job"}:
                slot = "occupation"
            elif pred in {"title", "role"}:
                slot = "title"
                
            if slot:
                candidates[slot][obj] += 1
                
        # Filter low freq
        return {k: v for k, v in candidates.items() if len(v) > 0}

    def _refine_vocab_with_llm(self, slot: str, counts: Counter) -> List[Dict[str, Any]]:
        # Take top 50 terms
        top_terms = counts.most_common(50)
        candidates_block = "\n".join([f"{term}: {count}" for term, count in top_terms])
        
        prompt = VOCAB_REFINEMENT_PROMPT.format(slot=slot, candidates_block=candidates_block)
        
        client = self._get_llm_client()
        messages = [{"role": "user", "content": prompt}]
        
        try:
            logger.info(f"Refining vocab for slot '{slot}'...")
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
            return data.get("groups", [])
        except Exception as e:
            logger.error(f"LLM Vocab Refinement failed: {e}")
            return []

    def _update_vocab_files(self, updates: Dict[str, List[Dict]], output_dir: Optional[Path] = None):
        target_vocab_path = (output_dir / "vocab.json") if output_dir else self.vocab_path
        target_aliases_path = (output_dir / "aliases.json") if output_dir else self.aliases_path
        
        # Load existing
        vocab = {}
        if target_vocab_path.exists():
            with open(target_vocab_path, "r") as f:
                vocab = json.load(f)
        
        aliases = {}
        if target_aliases_path.exists():
            with open(target_aliases_path, "r") as f:
                aliases = json.load(f)
                
        # Apply updates
        for slot, groups in updates.items():
            slot_vocab = vocab.setdefault(slot, {})
            slot_aliases = aliases.setdefault(slot, {})
            
            for group in groups:
                canon = group["canonical"]
                group_aliases = group.get("aliases", [])
                
                # Update vocab entry
                if canon not in slot_vocab:
                    slot_vocab[canon] = {
                        "canonical": canon,
                        "aliases": group_aliases,
                        "evidence": group.get("evidence", "")
                    }
                else:
                    # Merge aliases
                    existing_aliases = set(slot_vocab[canon].get("aliases", []))
                    existing_aliases.update(group_aliases)
                    slot_vocab[canon]["aliases"] = list(existing_aliases)
                    
                # Update flat aliases map
                for a in group_aliases:
                    slot_aliases[a] = canon
                    
        # Save
        with open(target_vocab_path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
            
        with open(target_aliases_path, "w") as f:
            json.dump(aliases, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Updated vocab and aliases in {target_vocab_path.parent}")
