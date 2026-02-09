import json
import sys
import os

def load_dataset(path):
    print(f"Loading {path}...")
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                data[row['id']] = row
            except:
                continue
    return data

def compare_datasets(subset_path, full_path):
    subset_data = load_dataset(subset_path)
    full_data = load_dataset(full_path)
    
    print(f"Subset size: {len(subset_data)}")
    print(f"Full set size: {len(full_data)}")
    
    missing_in_full = 0
    diff_gold_sp = 0
    diff_paragraphs = 0
    checked_count = 0
    
    # Specific case to check
    target_id = "2hop__145282_21711"
    
    for doc_id, subset_row in subset_data.items():
        if doc_id not in full_data:
            missing_in_full += 1
            continue
            
        full_row = full_data[doc_id]
        checked_count += 1
        
        # Check Paragraphs count
        if len(subset_row.get('paragraphs', [])) != len(full_row.get('paragraphs', [])):
             diff_paragraphs += 1

        # Check Gold SP
        # Normalize gold_sp to compare (list of lists)
        subset_gold = json.dumps(subset_row.get('paragraphs', []), sort_keys=True) # Wait, gold_sp is different.
        # Actually gold_sp is a list of [title, index] usually, but let's look at the fields.
        
        # Let's compare raw paragraphs length first as a proxy for "did we cut context?"
        # And compare question_decomposition
        
        # The user specifically asked about gold_sp
        # In MuSiQue, the support paragraphs are often marked in the paragraphs list strictly speaking? 
        # Or is there a separate paragraphs list?
        # Let's look at the structure of the rows.
        
        # Usually gold_sp is inferred from is_supporting field in paragraphs?
        # Or is it a separate field? 
        # Let's just compare the whole object for that ID for key fields.
        
        if doc_id == target_id:
            print(f"\n[Target Case Analysis: {doc_id}]")
            print("Subset Row Keys:", subset_row.keys())
            print("Full Row Keys:", full_row.keys())
            
            # Print paragraphs with is_supporting=True
            print("\nSubset Supporting Paragraphs:")
            for p in subset_row.get('paragraphs', []):
                if p.get('is_supporting'):
                    print(f" - {p.get('title')} (idx: {p.get('idx')})")
            
            print("\nFull Set Supporting Paragraphs:")
            for p in full_row.get('paragraphs', []):
                if p.get('is_supporting'):
                    print(f" - {p.get('title')} (idx: {p.get('idx')})")

    print("\n[Comparison Summary]")
    print(f"Checked IDs: {checked_count}")
    print(f"IDs missing in full set: {missing_in_full}")
    print(f"Samples with different paragraph counts: {diff_paragraphs}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 compare_musique_gold.py <subset_file> <full_file>")
        sys.exit(1)
        
    compare_datasets(sys.argv[1], sys.argv[2])
