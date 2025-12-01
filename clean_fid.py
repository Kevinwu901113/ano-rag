import csv
import sys
import re

def _strip_reasoning(text: str) -> str:
    output = text or ""
    # Strip standard <think> tags if present
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            output = output[:start]
            break
        output = output[:start] + output[end + len("</think>") :]
    
    cleaned = output.strip()
    return cleaned or "Insufficient evidence"

def _enforce_short_answer(text: str) -> str:
    if not text:
        return "Insufficient evidence"
    
    # 1. Handle common conversational prefixes (case-insensitive)
    conversational_patterns = [
        r"^(okay|ok|so|well|hmm|let's see|let me see|let me look|let me try|i need to|the user is asking|the question is asking|first, i need to|let me go through|let me check|determine)[\.,]?",
        r"^based on the (provided )?context,?",
        r"^the answer is",
        r"^the answer appears to be",
        r"^according to the context,?",
        r"^it seems that",
        r"^about",
        r"^(i need to|i must|i should)",
        r"^(from the|in the) (given|provided)? ?context",
        r"^(to find|to determine|to answer|to figure out)",
        r"^what the answer is",
        r"^the question is asking",
        r"^the user provided",
        r"^his job, right\?",
        r"^check the context provided",
        r"^([a-zA-Z0-9' \.-]+)'s occupation is", 
        r"^the occupation of [a-zA-Z0-9' \.-]+ is",
    ]
    
    cleaned = text.strip()
    
    # Iteratively remove prefixes
    for _ in range(10):
        original = cleaned
        for pattern in conversational_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Strip Markdown bold/italic markers
        for _ in range(5):
            prev = cleaned
            if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) >= 4:
                cleaned = cleaned[2:-2].strip()
            elif cleaned.startswith("*") and cleaned.endswith("*") and len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
            if cleaned.startswith("**"):
                cleaned = cleaned[2:].strip()
            if cleaned.endswith("**"):
                cleaned = cleaned[:-2].strip()
            if cleaned == prev:
                break
            
        if cleaned == original:
            break

    # 2. Handle "Answer:" markers
    if "Answer:" in cleaned:
        parts = cleaned.split("Answer:")
        cleaned = parts[-1].strip()
    elif "answer:" in cleaned.lower():
        parts = re.split(r"answer:", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1].strip()

    lines = [L.strip() for L in cleaned.splitlines() if L.strip()]
    if not lines:
        return "Insufficient evidence"
    
    first_line = lines[0]
    
    if len(first_line) > 100:
        match = re.search(r"\b(is|was) (a|an|the) (.+?)(\.|$)", first_line, re.IGNORECASE)
        if match:
             candidate = match.group(3).strip()
             if len(candidate) < 50:
                 first_line = candidate

    cleaned = first_line
    if len(cleaned) >= 2 and ((cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'"))):
        cleaned = cleaned[1:-1].strip()
        
    if cleaned.endswith(".") and not cleaned.endswith("Inc.") and not cleaned.endswith("St."):
        cleaned = cleaned[:-1].strip()

    final_answer = cleaned
    
    if "insufficient evidence" in final_answer.lower():
        if len(final_answer) < 30 and "insufficient evidence" in final_answer.lower():
             return "Insufficient evidence"

    return final_answer or "Insufficient evidence"


def clean_fid_file(input_path, output_path):
    print(f"Cleaning {input_path} -> {output_path}")
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                query = row[0]
                raw_answer = row[1]
                # First strip reasoning/think tags
                stripped = _strip_reasoning(raw_answer)
                # Then enforce short answer format
                cleaned_answer = _enforce_short_answer(stripped)
                rows.append([query, cleaned_answer])
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_fid.py <input_tsv> <output_tsv>")
        sys.exit(1)
    
    clean_fid_file(sys.argv[1], sys.argv[2])
