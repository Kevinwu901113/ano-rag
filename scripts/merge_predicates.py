#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES, PRED_SYNONYM_SETS

PREDICATES_JSON_PATH = Path(__file__).resolve().parents[1] / "relrag/schema/predicates.json"

def main():
    input_file = "candidate_predicates.jsonl"
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found.")
        return

    # Load current predicates.json
    current_preds: Dict[str, List[str]] = {}
    if PREDICATES_JSON_PATH.exists():
        with open(PREDICATES_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                if isinstance(content, dict):
                    current_preds = content
                elif isinstance(content, list):
                    # Convert list to dict (no synonyms initially)
                    current_preds = {p: [] for p in content}
            except json.JSONDecodeError:
                pass

    # Build lookup for existing canonicals and synonyms
    # We use PRED_SYNONYM_SETS from code as the baseline truth
    canonical_map: Dict[str, str] = {} # synonym -> canonical
    
    # 1. Add code-defined predicates
    for canon, synonyms in PRED_SYNONYM_SETS.items():
        canonical_map[canon] = canon
        for syn in synonyms:
            canonical_map[syn] = canon
            
    # 2. Add/Override with predicates.json
    for canon, synonyms in current_preds.items():
        canonical_map[canon] = canon
        for syn in synonyms:
            canonical_map[syn] = canon

    # Statistics
    added_new = 0
    added_synonyms = 0
    skipped = 0
    
    # Process candidates
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue
                
            proposed = item.get("proposed_pred", "").strip().lower()
            if not proposed:
                continue
                
            surface_forms = [s.strip().lower() for s in item.get("surface_forms", []) if s.strip()]
            
            # Check if proposed is already mapped
            if proposed in canonical_map:
                existing_canon = canonical_map[proposed]
                # It's a synonym or existing canonical
                # Add new surface forms to the existing canonical
                if existing_canon not in current_preds:
                    current_preds[existing_canon] = []
                
                changed = False
                for sf in surface_forms:
                    if sf not in canonical_map: # New synonym
                        current_preds[existing_canon].append(sf)
                        canonical_map[sf] = existing_canon
                        changed = True
                
                if changed:
                    added_synonyms += 1
                else:
                    skipped += 1
            else:
                # Completely new predicate
                # But wait, maybe it maps to an existing one?
                # The LLM prompt asked to propose NEW ones, but it might hallucinate existing ones with new names
                # e.g. "writer_of" -> "authored_by"
                
                # Heuristic: check if any surface form maps to an existing canonical
                mapped_canon = None
                for sf in surface_forms:
                    if sf in canonical_map:
                        mapped_canon = canonical_map[sf]
                        break
                
                if mapped_canon:
                    # Treat as synonym for mapped_canon
                    if mapped_canon not in current_preds:
                        current_preds[mapped_canon] = []
                    
                    for sf in surface_forms:
                        if sf not in canonical_map:
                            current_preds[mapped_canon].append(sf)
                            canonical_map[sf] = mapped_canon
                    
                    # Also map the proposed name itself as a synonym
                    if proposed not in canonical_map:
                        current_preds[mapped_canon].append(proposed)
                        canonical_map[proposed] = mapped_canon
                        
                    added_synonyms += 1
                else:
                    # It's a new canonical
                    if proposed not in current_preds:
                        current_preds[proposed] = []
                        added_new += 1
                    
                    for sf in surface_forms:
                        if sf not in current_preds[proposed]:
                            current_preds[proposed].append(sf)
                            canonical_map[sf] = proposed

    # Save updated predicates.json
    with open(PREDICATES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(current_preds, f, indent=2, sort_keys=True)
        
    print(f"Merge complete.")
    print(f"Added {added_new} new predicates.")
    print(f"Added synonyms to {added_synonyms} existing predicates.")
    print(f"Skipped {skipped} redundant entries.")
    print(f"Total predicates: {len(current_preds)}")

if __name__ == "__main__":
    main()
