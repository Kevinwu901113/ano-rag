import re

def strip_reasoning(text: str) -> str:
    """
    Strips <think>...</think> blocks and other reasoning artifacts.
    """
    output = text or ""
    # Strip standard <think> tags if present
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            # Unclosed tag, strip everything after start
            output = output[:start]
            break
        output = output[:start] + output[end + len("</think>") :]
    
    cleaned = output.strip()
    return cleaned or "Insufficient evidence"


def enforce_short_answer(text: str) -> str:
    """
    Clean up the LLM output to ensure it's just the answer.
    Handles conversational fillers and reasoning that might have slipped through.
    """
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
    ]
    
    cleaned = text.strip()
    
    # Iteratively remove prefixes until no more matches found
    while True:
        original = cleaned
        for pattern in conversational_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == original:
            break
            
    # 2. Handle "Answer:" markers if present
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
    
    # If the first line is still very long, it might be a sentence.
    if len(first_line) > 100:
        # Emergency: try to find "is a" / "was a" again
        match = re.search(r"\b(is|was) (a|an|the) (.+?)(\.|$)", first_line, re.IGNORECASE)
        if match:
             candidate = match.group(3).strip()
             if len(candidate) < 50:
                 first_line = candidate

    cleaned = first_line
    # Remove surrounding quotes if present
    if len(cleaned) >= 2 and ((cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'"))):
        cleaned = cleaned[1:-1].strip()
        
    # Remove trailing period if it looks like a sentence end
    if cleaned.endswith(".") and not cleaned.endswith("Inc.") and not cleaned.endswith("St."):
        cleaned = cleaned[:-1].strip()

    # 5. Check for "Insufficient evidence" variations
    if len(cleaned) < 30 and "insufficient evidence" in cleaned.lower():
         return "Insufficient evidence"

    return cleaned or "Insufficient evidence"

def clean_model_answer(text: str) -> str:
    """
    Full pipeline: strip reasoning -> enforce short answer.
    """
    no_reasoning = strip_reasoning(text)
    return enforce_short_answer(no_reasoning)
