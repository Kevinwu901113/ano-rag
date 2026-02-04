from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

OUTPUT_ROOT = Path("result/ultradomain/lightrag_protocol_v1_mix_legal")
RUN_META_DIR = OUTPUT_ROOT / "run_meta"
CHUNKS_DIR = OUTPUT_ROOT / "chunks"
QUESTIONS_DIR = OUTPUT_ROOT / "questions"
ANSWERS_DIR = OUTPUT_ROOT / "answers"
JUDGE_DIR = OUTPUT_ROOT / "judge"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
INDEX_DIR = OUTPUT_ROOT / "indexes"

TOKENIZER_ID = "deepseek-ai/DeepSeek-V3.2"

DOMAIN_LABELS = {
    "mix": "Mix",
    "legal": "Legal",
}

_DOMAIN_ALIASES = {
    "mix": "mix",
    "mixed": "mix",
    "legal": "legal",
}

_TOKENIZER = None


def ensure_dirs() -> None:
    for path in (
        OUTPUT_ROOT,
        RUN_META_DIR,
        CHUNKS_DIR,
        QUESTIONS_DIR,
        ANSWERS_DIR,
        JUDGE_DIR,
        SUMMARY_DIR,
        INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def normalize_domain(label: str) -> Optional[str]:
    if not label:
        return None
    key = _DOMAIN_ALIASES.get(str(label).strip().lower())
    if key in DOMAIN_LABELS:
        return key
    return None


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def chunk_note_id(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}#{chunk_id}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return iter(())
    def _iter() -> Iterator[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    return _iter()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def get_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    fallback = os.environ.get("OPENAI_API_KEY")
    if fallback:
        return fallback
    raise RuntimeError(f"Missing API key: set {env_name} (or OPENAI_API_KEY).")


def get_git_commit() -> Optional[str]:
    try:
        raw = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return raw.decode("utf-8").strip()
    except Exception:
        return None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    from transformers import AutoTokenizer  # type: ignore

    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            trust_remote_code=True,
            use_fast=False,
        )
    return _TOKENIZER


def encode_with_offsets(text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
    tokenizer = get_tokenizer()
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded.get("input_ids") or [])
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            return input_ids, None
        return input_ids, [(int(s), int(e)) for s, e in offsets]
    except Exception:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        return list(input_ids), None


def decode_tokens(token_ids: List[int]) -> str:
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def assemble_budgeted_items(
    items: List[Dict[str, Any]],
    budget_tokens: int,
) -> Tuple[List[Dict[str, Any]], int]:
    selected: List[Dict[str, Any]] = []
    used = 0
    for item in items:
        text = item.get("text") or item.get("evidence") or item.get("canonical") or ""
        if not text:
            continue
        token_count = item.get("token_count")
        if token_count is None:
            token_count = count_tokens(str(text))
        token_count = int(token_count)
        if used + token_count > budget_tokens:
            break
        cloned = dict(item)
        cloned["token_count"] = token_count
        selected.append(cloned)
        used += token_count
    return selected, used


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : idx + 1]
                return json.loads(candidate)
    raise ValueError("unterminated JSON object")


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*", "", cleaned).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")].strip()
    return cleaned


def extract_json_list(text: str) -> List[Any]:
    cleaned = _strip_code_fence(text)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("no JSON list found")
    depth = 0
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : idx + 1]
                return json.loads(candidate)
    raise ValueError("unterminated JSON list")

