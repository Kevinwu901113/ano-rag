
import sys
import os
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from adapters.mirage import iter_docs_and_chunks
from validators.note_validator import validate_and_normalize, _looks_like_occupation

def verify_doc_ids():
    print("\n--- Verifying Doc ID Uniqueness (MIRAGE) ---")
    data_dir = "data/mirage_sample"
    
    # Track ID -> set of titles to detect collisions
    id_map = {}
    
    try:
        # iter_docs_and_chunks yields (doc, chunk) pairs
        # The same doc is yielded multiple times (once per chunk)
        for doc, _ in iter_docs_and_chunks(data_dir):
            did = doc["doc_id"]
            title = doc["title"]
            
            if did not in id_map:
                id_map[did] = set()
            id_map[did].add(title)
                
        # Analyze results
        collisions = 0
        for did, titles in id_map.items():
            if len(titles) > 1:
                print(f"FAIL: ID collision for {did}. Shared by titles: {titles}")
                collisions += 1
                
        # Check specific John Mayne entries
        john_mayne_ids = {did: titles for did, titles in id_map.items() 
                          if any("John Mayne" in t or "John Dawson Mayne" in t for t in titles)}
        
        print(f"\nFound {len(john_mayne_ids)} John Mayne related distinct IDs:")
        for did, titles in john_mayne_ids.items():
            print(f"  ID: {did}")
            print(f"  Titles: {titles}")
            
        if collisions == 0:
            print("\nSUCCESS: No Doc ID collisions detected.")
        else:
            print(f"\nFAIL: {collisions} ID collisions detected.")
            
    except Exception as e:
        print(f"Error during doc ID verification: {e}")

def verify_validator_logic():
    print("\n--- Verifying Note Validator Logic (Occupation Detection) ---")
    
    # Simulate a note that lacks an explicit predicate but has definitional evidence
    # This simulates the "American politician" case for Kathy Saltzman
    
    test_cases = [
        {
            "name": "Kathy Saltzman (Politician)",
            "note": {
                "subj": "Kathy Saltzman",
                "pred": None,  # Missing predicate
                "obj": "politician",
                "evidence": "Kathy L. Saltzman (born June 4, 1955) is a Minnesota politician",
                "subj_type": "PERSON",
                "obj_type": "CONCEPT",
                "meta": {}
            },
            "expected_pred": "occupation"
        },
        {
            "name": "Eleanor Davis (Cartoonist)",
            "note": {
                "subj": "Eleanor Davis",
                "pred": None,
                "obj": "American cartoonist",
                "evidence": "Eleanor McCutcheon Davis is an American cartoonist and illustrator",
                "subj_type": "PERSON",
                "obj_type": "CONCEPT",
                "meta": {}
            },
            "expected_pred": "occupation"
        },
        {
            "name": "Irrelevant Text",
            "note": {
                "subj": "John Doe",
                "pred": None,
                "obj": "something",
                "evidence": "John Doe went to the store yesterday.",
                "subj_type": "PERSON",
                "obj_type": "CONCEPT",
                "meta": {}
            },
            "expected_pred": None # Should remain None or not be "occupation"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        note = case['note']
        
        # Manually construct list for validate_and_normalize
        raw_text = json.dumps([note])
        
        result = validate_and_normalize(raw_text, "test_doc", "test_chunk")
        valid_notes = result.get("valid_notes", [])
        
        if not valid_notes:
            validated_pred = None
            if case['expected_pred'] is None:
                print(f"SUCCESS: Note correctly filtered out/ignored.")
            else:
                print(f"FAIL: Note was filtered out, expected {case['expected_pred']}")
                print(f"Stats: {result.get('stats')}")
        else:
            validated_note = valid_notes[0]
            validated_pred = validated_note.get("pred")
            
            print(f"  Input Predicate: {note.get('pred')}")
            print(f"  Output Predicate: {validated_pred}")
            
            if validated_pred == case['expected_pred']:
                 print("SUCCESS: Predicate correctly identified.")
            elif case['expected_pred'] is None:
                if validated_pred != "occupation":
                     print("SUCCESS: Correctly ignored irrelevant text (predicate not occupation).")
                else:
                     print(f"FAIL: Unexpectedly identified as occupation.")
            else:
                 print(f"FAIL: Expected {case['expected_pred']}, got {validated_pred}")


def verify_generated_notes():
    print("\n--- Verifying Generated Notes (End-to-End) ---")
    notes_path = "result/080-mirage/notes.jsonl"
    if not os.path.exists(notes_path):
        print(f"FAIL: Notes file not found at {notes_path}")
        return

    found_kathy = False
    found_eleanor = False

    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            note = json.loads(line)
            subj = note.get("subj")
            pred = note.get("pred")
            obj = note.get("obj")

            if subj == "Kathy Saltzman" and pred == "occupation" and obj == "politician":
                found_kathy = True
                print(f"SUCCESS: Found Kathy Saltzman occupation note: {note['evidence']}")
            
            if subj == "Eleanor Davis" and pred == "occupation" and obj == "cartoonist":
                found_eleanor = True
                print(f"SUCCESS: Found Eleanor Davis occupation note: {note['evidence']}")

    if found_kathy and found_eleanor:
        print("SUCCESS: All target notes found in generated output.")
    else:
        print(f"FAIL: Missing notes. Kathy: {found_kathy}, Eleanor: {found_eleanor}")

if __name__ == "__main__":
    verify_doc_ids()
    verify_validator_logic()
    verify_generated_notes()
