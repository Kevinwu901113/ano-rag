import json
import os
import re
import unicodedata
from typing import List, Dict
from retriever.scorer import subject_match as scorer_subject_match

# Mocking the BIND and normalization functions from retriever/operators.py to trace execution
# We will load the actual index files to see real data.

def _normalize_alias_query(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\([^)]*\)", "", value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[\W_]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()

def _loose_match(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False
    tokens = [t for t in needle.split() if len(t) > 3]
    return any(t in hay for t in tokens)

class MockIndexes:
    def __init__(self, directory: str):
        print(f"Loading indexes from {directory}...")
        
        alias_index_path = os.path.join(directory, "entity_alias_index.json")
        if os.path.exists(alias_index_path):
            with open(alias_index_path, "r", encoding="utf-8") as handle:
                raw_alias = json.load(handle)
                self.alias_to_entities = {
                    (alias or "").lower(): values for alias, values in raw_alias.items()
                }
            print(f"Loaded {len(self.alias_to_entities)} alias entries.")
        else:
            self.alias_to_entities = {}
            print("No alias index found.")

        with open(os.path.join(directory, "entity_to_notes.json"), "r", encoding="utf-8") as handle:
            self.entity_to_notes = json.load(handle)
        print(f"Loaded {len(self.entity_to_notes)} entities.")


def trace_bind(indexes: MockIndexes, alias: str, limit: int = 50) -> List[str]:
    print(f"\n--- Tracing BIND for alias: '{alias}' ---")
    matches: List[str] = []
    target = _normalize_alias_query(alias)
    print(f"Normalized target: '{target}'")
    
    if not target:
        return matches

    # 1. Alias Exact Match
    alias_hits = indexes.alias_to_entities.get(target, [])
    if alias_hits:
        print(f"[Match] Alias exact match found: {alias_hits}")
        for entity in alias_hits:
            if entity not in matches:
                matches.append(entity)
                if len(matches) >= limit: return matches
    else:
        print("[No Match] Alias exact match failed.")

    # 2. Alias Normalized Exact Match
    print("Checking normalized alias keys...")
    for alias_key, entities in indexes.alias_to_entities.items():
        norm_key = _normalize_alias_query(alias_key)
        if norm_key == target and alias_key != target and not alias_hits:
            print(f"[Match] Normalized alias key '{alias_key}' (norm: '{norm_key}') matches target.")
            for entity in entities:
                if entity not in matches:
                    matches.append(entity)
                    print(f"  -> Added entity: {entity}")
                    if len(matches) >= limit: return matches

    # 3. Alias Contains Match (only if no matches yet)
    if not matches and indexes.alias_to_entities:
        print("Checking alias containment (fallback)...")
        for alias_key, entities in indexes.alias_to_entities.items():
            if alias_key == target: continue
            if target in alias_key or alias_key in target:
                print(f"[Match] Partial match with alias '{alias_key}'.")
                for entity in entities:
                    if entity in matches: continue
                    matches.append(entity)
                    print(f"  -> Added entity: {entity}")
                    if len(matches) >= limit: return matches

    # 4. Entity Name Matching (Exact & Loose)
    print("Checking entity names...")
    exact_norm: List[str] = []
    loose_norm: List[str] = []
    
    # We'll limit the search to relevant entities to avoid flooding logs, but simulate full scan
    # For this trace, we specifically look for "John Mayne" and "John Dawson Mayne" related keys
    
    candidates_to_watch = ["John Mayne", "John Dawson Mayne"]
    
    for entity in indexes.entity_to_notes.keys():
        norm_name = _normalize_alias_query(entity)
        if not norm_name or len(norm_name) < 3:
            continue
            
        is_watched = any(c.lower() in entity.lower() for c in candidates_to_watch)
        
        if norm_name == target:
            exact_norm.append(entity)
            if is_watched: print(f"[Match] Entity exact norm match: '{entity}'")
        elif target in norm_name or norm_name in target:
            loose_norm.append(entity)
            if is_watched: print(f"[Match] Entity partial norm match: '{entity}' (norm: '{norm_name}')")
        elif _loose_match(target, norm_name):
            loose_norm.append(entity)
            if is_watched: print(f"[Match] Entity loose token match: '{entity}'")

    for bucket_name, bucket in [("Exact", exact_norm), ("Loose", loose_norm)]:
        for entity in bucket:
            if entity in matches: continue
            matches.append(entity)
            # print(f"  -> Added {bucket_name} entity: {entity}") # Too noisy for all
            if len(matches) >= limit: return matches

    return matches

def subject_match(subject: str | None, seeds: List[str]) -> float:
    return scorer_subject_match(subject, seeds, None)

def main():
    indexes_dir = "result/080-mirage/indexes"
    
    print("\n--- Testing Subject Match Scoring ---")
    seeds = ["John Mayne"]
    candidates = ["John Mayne", "John Dawson Mayne", "John", "Mayne", "Kathy Saltzman"]
    
    for cand in candidates:
        score = subject_match(cand, seeds)
        print(f"Subject: '{cand}' | Seed: {seeds} | Score: {score}")

    # Re-verify why John Dawson Mayne might appear in results if not bound
    # Check if there are notes with "John Dawson Mayne" as subject
    print("\n--- Checking Notes for 'John Dawson Mayne' ---")
    try:
        if not os.path.exists(indexes_dir):
            indexes_dir = "indexes"
        
        with open(os.path.join(indexes_dir, "entity_to_notes.json"), "r", encoding="utf-8") as handle:
            entity_to_notes = json.load(handle)
            
        jdm_notes = entity_to_notes.get("John Dawson Mayne", [])
        print(f"Notes for 'John Dawson Mayne': {len(jdm_notes)}")
        
        jm_notes = entity_to_notes.get("John Mayne", [])
        print(f"Notes for 'John Mayne': {len(jm_notes)}")
        print(f"Content of 'John Mayne' notes: {jm_notes}")
        
    except Exception as e:
        print(f"Error checking notes: {e}")



if __name__ == "__main__":
    main()
