import json
import re


def clean_control_characters(text: str) -> str:
    """Remove most control characters but keep common whitespace."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    cleaned = cleaned.replace('\u0000', '').replace('\u0001', '').replace('\u0002', '')
    return cleaned


def extract_json_from_response(response: str) -> str:
    """Try to extract a JSON string from a larger LLM response."""
    if not response or not str(response).strip():
        return ""

    text = clean_control_characters(str(response).strip())

    # If the whole text is JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Remove common markdown code block markers
    if text.startswith('```') and text.endswith('```'):
        text_block = re.sub(r'^```(?:json)?', '', text[:-3], flags=re.IGNORECASE).strip()
        try:
            json.loads(text_block)
            return text_block
        except Exception:
            pass

    # Search for JSON inside markdown code block
    code_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if code_match:
        candidate = clean_control_characters(code_match.group(1).strip())
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # Search for JSON object or array in the text
    brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if brace_match:
        candidate = clean_control_characters(brace_match.group(1))
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return ""
