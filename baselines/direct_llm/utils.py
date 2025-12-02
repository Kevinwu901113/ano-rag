
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
