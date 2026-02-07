from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from relrag.config.config_loader import config as config_loader


def _nested_get(mapping: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def ultradomain_config() -> Dict[str, Any]:
    try:
        cfg = config_loader.load_config()
    except Exception:
        return {}
    section = cfg.get("ultradomain")
    return section if isinstance(section, dict) else {}


def ultradomain_get(path: str, default: Any = None) -> Any:
    return _nested_get(ultradomain_config(), path, default)


def _resolve_output_root() -> Path:
    value = ultradomain_get("protocol.output_root")
    if isinstance(value, str) and value.strip():
        return Path(value.strip())
    return Path("result/ultradomain/lightrag_protocol_v1_mix_legal")


OUTPUT_ROOT = _resolve_output_root()
RUN_META_DIR = OUTPUT_ROOT / "run_meta"
CHUNKS_DIR = OUTPUT_ROOT / "chunks"
QUESTIONS_DIR = OUTPUT_ROOT / "questions"
ANSWERS_DIR = OUTPUT_ROOT / "answers"
JUDGE_DIR = OUTPUT_ROOT / "judge"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
INDEX_DIR = OUTPUT_ROOT / "indexes"

DEFAULT_TOKENIZER_ID = "deepseek-ai/DeepSeek-V3.2"


def _resolve_preferred_tokenizer_id() -> str:
    preferred = ultradomain_get("tokenizer.model")
    if isinstance(preferred, str) and preferred.strip():
        return preferred.strip()
    llm_model = ultradomain_get("llm.model")
    if isinstance(llm_model, str) and llm_model.strip():
        return llm_model.strip()
    try:
        cfg = config_loader.load_config()
        vllm_model = ((cfg.get("vllm") or {}).get("model") or "").strip()
        if vllm_model:
            return vllm_model
    except Exception:
        pass
    return DEFAULT_TOKENIZER_ID


TOKENIZER_ID = _resolve_preferred_tokenizer_id()

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
_TOKENIZER_SOURCE: Optional[str] = None


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
    global _TOKENIZER_SOURCE
    if _TOKENIZER is not None:
        return _TOKENIZER
    from transformers import AutoTokenizer  # type: ignore

    local_only = bool(ultradomain_get("tokenizer.local_only", True))
    trust_remote_code = bool(ultradomain_get("tokenizer.trust_remote_code", True))
    configured_local_path = ultradomain_get("tokenizer.local_path")
    fallback_tokenizer = ultradomain_get("tokenizer.fallback_model", DEFAULT_TOKENIZER_ID)
    env_tokenizer_path = os.environ.get("ULTRADOMAIN_TOKENIZER_PATH")

    candidates: List[str] = []
    for value in (
        configured_local_path,
        env_tokenizer_path,
        TOKENIZER_ID,
        fallback_tokenizer,
        DEFAULT_TOKENIZER_ID,
    ):
        text = str(value or "").strip()
        if not text or text in candidates:
            continue
        candidates.append(text)

    errors: List[str] = []
    for candidate in candidates:
        for use_fast in (True, False):
            # Always try local cache/path first to avoid network dependency.
            try:
                _TOKENIZER = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                    local_files_only=True,
                )
                _TOKENIZER_SOURCE = candidate
                return _TOKENIZER
            except Exception as exc:
                errors.append(f"{candidate} [fast={use_fast}, local_only=True]: {exc}")

            if local_only:
                continue
            try:
                _TOKENIZER = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                )
                _TOKENIZER_SOURCE = candidate
                return _TOKENIZER
            except Exception as exc:
                errors.append(f"{candidate} [fast={use_fast}, local_only=False]: {exc}")

    raise RuntimeError(
        "Failed to load tokenizer from local-first candidates. "
        f"candidates={candidates}. "
        "Set ultradomain.tokenizer.local_path or env ULTRADOMAIN_TOKENIZER_PATH "
        "to the local model directory used by vLLM."
    )


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
