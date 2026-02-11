from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

QWEN_CHAT_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_CHAT_MODEL = "qwen3-30b-a3b"
EMBED_BASE_URL = "http://127.0.0.1:8001/v1"
EMBED_MODEL = "qwen3-embedding"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

REFUSAL_KEYWORDS = (
    "insufficient evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "i don't know",
    "unknown",
)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str
    model: str
    api_key: str


def _resolve_deepseek_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if key:
        return key
    raise RuntimeError(
        "Missing DeepSeek API key. Set OPENAI_API_KEY or DEEPSEEK_API_KEY."
    )


def resolve_llm_backend(name: str) -> BackendConfig:
    backend = str(name).strip().lower()
    if backend == "qwen":
        return BackendConfig(
            name="qwen",
            base_url=QWEN_CHAT_BASE_URL,
            model=QWEN_CHAT_MODEL,
            api_key="EMPTY",
        )
    if backend == "deepseek":
        return BackendConfig(
            name="deepseek",
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            api_key=_resolve_deepseek_api_key(),
        )
    raise ValueError(f"Unsupported llm backend: {name}")


def load_json(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list at {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_qa(path: Path, limit: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def ensure_dataset(dataset: str) -> str:
    value = str(dataset).strip().lower()
    if value not in {"hotpotqa", "musique", "2wiki"}:
        raise ValueError("dataset must be one of: hotpotqa, musique, 2wiki")
    return value


def output_pred_path(output_root: Path, method: str, dataset: str, backend: str) -> Path:
    out_dir = output_root / method / dataset / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "pred.jsonl"


def write_pred_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def looks_refusal(text: str) -> bool:
    norm = str(text or "").strip().lower()
    return any(key in norm for key in REFUSAL_KEYWORDS)
