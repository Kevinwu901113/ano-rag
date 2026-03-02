from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

QWEN_CHAT_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_CHAT_MODEL = "qwen3-30b-a3b"
EMBED_BASE_URL = "http://127.0.0.1:8001/v1"
EMBED_MODEL = "qwen3-embedding"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

REFUSAL_KEYWORDS = (
    "insufficient evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "i don't know",
    "unknown",
)

_REF_SECTION_RE = re.compile(r"(?is)\n+#{1,6}\s*references\b.*$")
_LINE_REF_HEADER_RE = re.compile(r"(?i)^#{0,6}\s*references\s*:?\s*$")
_LINE_CITATION_RE = re.compile(r"^\s*[-*]?\s*\[\d+\]\s+")
_LINE_HEADER_RE = re.compile(r"^\s*#{1,6}\s*")
_LINE_BULLET_RE = re.compile(r"^\s*[-*+]\s*")
_ANSWER_PREFIX_RE = re.compile(r"(?i)^(answer|final answer)\s*[:：-]\s*")
_TRAILING_CITATION_RE = re.compile(r"\s*\[\d+\]\s*$")
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_DANGLING_RE = re.compile(r"(?is)<think>.*$")
_FINAL_LINE_RE = re.compile(r"(?im)^\s*FINAL\s*[:：-]\s*(.+?)\s*$")
DEEPSEEK_MIN_ANSWER_TOKENS = 512

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DIR = _REPO_ROOT / "relrag" / "prompt"
_ALIGNED_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "system_prompt_openai.txt"
_ALIGNED_ANSWER_PROMPT_PATH = _PROMPT_DIR / "answerer_openai.txt"
_ALIGNED_FINAL_INSTRUCTION_PATH = _PROMPT_DIR / "final_instruction.txt"
_ALIGNED_LABEL_INSTRUCTION = "If you can answer, output the canonical label only."
_PROMPT_CACHE: Dict[str, str] = {}


def _load_deepseek_api_key_from_relrag_config() -> str:
    candidates: List[Path] = []
    config_env = str(os.getenv("ANO_RAG_CONFIG") or "").strip()
    if config_env:
        candidates.append(Path(config_env).expanduser())
    candidates.append(_REPO_ROOT / "relrag" / "config" / "config.yaml")

    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml  # lazy import

            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                continue
            openai_cfg = payload.get("openai") if isinstance(payload.get("openai"), dict) else {}
            key = str(openai_cfg.get("api_key") or "").strip()
            if key and not key.startswith("${"):
                return key
        except Exception:
            continue
    return ""


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str
    model: str
    api_key: str


@dataclass
class CostRecord:
    index_time_ms: Optional[float]
    query_time_ms: Optional[float]
    query_retrieval_ms: Optional[float]
    query_reader_ms: Optional[float]
    llm_calls: int
    llm_retries: int
    prompt_tokens_total: Optional[int]
    completion_tokens_total: Optional[int]
    total_tokens: Optional[int]
    token_source: str
    token_unavailable_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_time_ms": self.index_time_ms,
            "query_time_ms": self.query_time_ms,
            "query_retrieval_ms": self.query_retrieval_ms,
            "query_reader_ms": self.query_reader_ms,
            "llm_calls": int(self.llm_calls),
            "llm_retries": int(self.llm_retries),
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens": self.total_tokens,
            "token_source": self.token_source,
            "token_unavailable_reason": self.token_unavailable_reason,
        }


def _resolve_deepseek_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if key:
        return key
    key = _load_deepseek_api_key_from_relrag_config()
    if key:
        return key
    raise RuntimeError(
        "Missing DeepSeek API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY "
        "or configure relrag/config/config.yaml -> openai.api_key."
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


def normalize_doc_pool(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = row.get("docs") or []
    if not isinstance(docs, list):
        raise ValueError("row.docs must be a list")
    normalized: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        if not title or not text:
            continue
        source_idx_raw = doc.get("source_idx")
        try:
            source_idx = int(source_idx_raw) if source_idx_raw is not None else idx
        except Exception:
            source_idx = idx
        normalized.append(
            {
                "id": str(doc.get("id") or f"qdoc_{idx + 1:04d}"),
                "title": title,
                "text": text,
                "is_supporting": (
                    None if doc.get("is_supporting") is None else bool(doc.get("is_supporting"))
                ),
                "source_idx": source_idx,
            }
        )
    normalized.sort(key=lambda item: (int(item.get("source_idx", 0)), str(item.get("id", ""))))
    return normalized


def normalize_chunk_pool(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = row.get("chunks") or []
    if not isinstance(chunks, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        title = str(chunk.get("title") or "").strip()
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        source_idx_raw = chunk.get("source_idx")
        try:
            source_idx = int(source_idx_raw) if source_idx_raw is not None else idx
        except Exception:
            source_idx = idx
        chunk_idx_raw = chunk.get("chunk_idx")
        try:
            chunk_idx = int(chunk_idx_raw) if chunk_idx_raw is not None else idx
        except Exception:
            chunk_idx = idx
        normalized.append(
            {
                "id": str(chunk.get("id") or f"qchunk_{idx + 1:04d}"),
                "title": title,
                "text": text,
                "is_supporting": (
                    None
                    if chunk.get("is_supporting") is None
                    else bool(chunk.get("is_supporting"))
                ),
                "source_idx": source_idx,
                "chunk_idx": chunk_idx,
                "parent_doc_id": str(chunk.get("parent_doc_id") or "").strip() or None,
            }
        )
    normalized.sort(
        key=lambda item: (
            int(item.get("source_idx", 0)),
            int(item.get("chunk_idx", 0)),
            str(item.get("id", "")),
        )
    )
    return normalized


def load_qa_with_docs(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows = load_qa(path, limit=limit)
    for row in rows:
        row["docs"] = normalize_doc_pool(row)
        row["chunks"] = normalize_chunk_pool(row)
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


def normalize_answer_for_eval(text: str, *, max_chars: int = 160) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    raw = _THINK_BLOCK_RE.sub("", raw)
    raw = _THINK_DANGLING_RE.sub("", raw)
    raw = raw.replace("<think>", "").replace("</think>", "").strip()
    final_matches = _FINAL_LINE_RE.findall(raw)
    if final_matches:
        raw = str(final_matches[-1]).strip()

    raw = _REF_SECTION_RE.sub("", raw).strip()
    lines: List[str] = []
    for row in raw.splitlines():
        line = str(row).strip()
        if not line:
            continue
        if _LINE_REF_HEADER_RE.match(line):
            break
        if _LINE_CITATION_RE.match(line):
            continue
        line = _LINE_HEADER_RE.sub("", line)
        line = _LINE_BULLET_RE.sub("", line)
        line = _ANSWER_PREFIX_RE.sub("", line).strip()
        line = _TRAILING_CITATION_RE.sub("", line).strip()
        if line:
            lines.append(line)

    if not lines:
        return ""

    answer = lines[0].strip().strip("`*_\"'")
    if not answer:
        return ""

    lower = answer.lower()
    if any(key in lower for key in REFUSAL_KEYWORDS):
        return "Insufficient evidence"

    answer = re.sub(r"(?i)^final\s*[:：-]\s*", "", answer).strip()

    if re.match(r"(?i)^yes\b", answer):
        return "Yes"
    if re.match(r"(?i)^no\b", answer):
        return "No"

    sentence_split = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)
    if sentence_split:
        answer = sentence_split[0].strip()

    answer = _ANSWER_PREFIX_RE.sub("", answer).strip()
    answer = _TRAILING_CITATION_RE.sub("", answer).strip()
    answer = answer.rstrip(" .;:")

    if len(answer) > max_chars:
        answer = answer[:max_chars].rstrip(" .;:")
    return answer


def _load_prompt_file(path: Path) -> str:
    key = str(path.resolve())
    cached = _PROMPT_CACHE.get(key)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[key] = text
    return text


def load_aligned_reader_system_prompt() -> str:
    return _load_prompt_file(_ALIGNED_SYSTEM_PROMPT_PATH)


def render_aligned_reader_prompt(
    *,
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
) -> str:
    try:
        from relrag.utils.openai_answer import build_answer_prompt as _build_answer_prompt

        evidences: List[Dict[str, Any]] = []
        for idx, row in enumerate(evidence_rows):
            rank = int(row.get("rank") or (idx + 1))
            rid = str(row.get("id") or f"doc_{rank}")
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            canonical = text
            if title and text:
                canonical = f"{title}: {text}"
            elif title and not text:
                canonical = title
            evidences.append(
                {
                    "note_id": rid,
                    "canonical": canonical,
                    "evidence": "",
                    "weak": False,
                }
            )
        return _build_answer_prompt(
            str(question or "").strip(),
            evidences,
            prompt_name="answerer_openai.txt",
        )
    except Exception:
        template = _load_prompt_file(_ALIGNED_ANSWER_PROMPT_PATH)
        strong_lines: List[str] = []
        for idx, row in enumerate(evidence_rows):
            rank = int(row.get("rank") or (idx + 1))
            rid = str(row.get("id") or f"doc_{rank}")
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            if title:
                strong_lines.append(f"{idx + 1}) [{rid}] {title}: {text}")
            else:
                strong_lines.append(f"{idx + 1}) [{rid}] {text}")
        strong_block = "\n".join(strong_lines) if strong_lines else "None"
        weak_block = "None"
        final_instruction = _load_prompt_file(_ALIGNED_FINAL_INSTRUCTION_PATH)
        return template.format(
            label_instruction=_ALIGNED_LABEL_INSTRUCTION,
            q=str(question or "").strip(),
            strong_block=strong_block,
            weak_block=weak_block,
            final_instruction=final_instruction,
        )


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_relrag_reader_policy(
    *,
    dataset: str,
    backend: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    fallback = {
        "temperature": 0.0,
        "answer_max_tokens": 256,
        "source": "fallback",
    }
    try:
        from relrag.config.config_loader import ConfigLoader, config as global_config
        from relrag.config.dataset_config import get_dataset_config, resolve_openai_config
    except Exception:
        return fallback

    try:
        cfg = ConfigLoader(config_path).load_config() if config_path else global_config.load_config()
    except Exception:
        return fallback

    dataset_key = str(dataset or "").strip()
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    backend_key = str(backend or "").strip().lower()

    if backend_key == "deepseek":
        openai_cfg = resolve_openai_config(cfg, dataset_cfg, overrides=None)
        return {
            "temperature": _coerce_float(openai_cfg.get("temperature"), 0.0),
            "answer_max_tokens": _coerce_int(openai_cfg.get("max_tokens"), 256),
            "source": "openai",
        }

    dataset_profiles = dataset_cfg.get("llm_profiles") if isinstance(dataset_cfg.get("llm_profiles"), dict) else {}
    dataset_generate = dataset_profiles.get("generate") if isinstance(dataset_profiles.get("generate"), dict) else {}

    root_profiles = cfg.get("llm_profiles") if isinstance(cfg.get("llm_profiles"), dict) else {}
    root_generate = root_profiles.get("generate") if isinstance(root_profiles.get("generate"), dict) else {}

    vllm_cfg = cfg.get("vllm") if isinstance(cfg.get("vllm"), dict) else {}
    temperature = dataset_generate.get(
        "temperature",
        root_generate.get("temperature", vllm_cfg.get("temperature", 0.0)),
    )
    max_tokens = dataset_generate.get(
        "max_tokens",
        root_generate.get("max_tokens", vllm_cfg.get("max_tokens", 256)),
    )
    return {
        "temperature": _coerce_float(temperature, 0.0),
        "answer_max_tokens": _coerce_int(max_tokens, 256),
        "source": "vllm_generate",
    }


def resolve_effective_reader_params(
    *,
    dataset: str,
    backend: str,
    answer_max_tokens: Optional[int],
    temperature: Optional[float],
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    policy = resolve_relrag_reader_policy(dataset=dataset, backend=backend, config_path=config_path)
    backend_key = str(backend or "").strip().lower()
    resolved_tokens = (
        _coerce_int(answer_max_tokens, 256)
        if answer_max_tokens is not None
        else _coerce_int(policy.get("answer_max_tokens"), 256)
    )
    if answer_max_tokens is None and backend_key == "deepseek":
        resolved_tokens = max(int(resolved_tokens), DEEPSEEK_MIN_ANSWER_TOKENS)
    resolved_temperature = (
        _coerce_float(temperature, 0.0)
        if temperature is not None
        else _coerce_float(policy.get("temperature"), 0.0)
    )
    return {
        "answer_max_tokens": int(resolved_tokens),
        "temperature": float(resolved_temperature),
        "source": str(policy.get("source") or "fallback"),
    }


def usage_prompt_completion(usage: Any) -> tuple[Optional[int], Optional[int]]:
    if usage is None:
        return None, None
    prompt_tokens = None
    completion_tokens = None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
    try:
        prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
    except Exception:
        prompt_tokens = None
    try:
        completion_tokens = int(completion_tokens) if completion_tokens is not None else None
    except Exception:
        completion_tokens = None
    return prompt_tokens, completion_tokens


def build_cost_record(
    *,
    index_time_ms: Optional[float],
    query_retrieval_ms: Optional[float],
    query_reader_ms: Optional[float],
    llm_calls: int,
    llm_retries: int,
    prompt_tokens_total: Optional[int],
    completion_tokens_total: Optional[int],
    token_source: str,
    token_unavailable_reason: Optional[str],
) -> CostRecord:
    query_time_ms = None
    if query_retrieval_ms is not None or query_reader_ms is not None:
        query_time_ms = float(query_retrieval_ms or 0.0) + float(query_reader_ms or 0.0)
    total_tokens = None
    if prompt_tokens_total is not None or completion_tokens_total is not None:
        total_tokens = int(prompt_tokens_total or 0) + int(completion_tokens_total or 0)
    return CostRecord(
        index_time_ms=(None if index_time_ms is None else float(index_time_ms)),
        query_time_ms=query_time_ms,
        query_retrieval_ms=(None if query_retrieval_ms is None else float(query_retrieval_ms)),
        query_reader_ms=(None if query_reader_ms is None else float(query_reader_ms)),
        llm_calls=int(llm_calls),
        llm_retries=int(llm_retries),
        prompt_tokens_total=(None if prompt_tokens_total is None else int(prompt_tokens_total)),
        completion_tokens_total=(None if completion_tokens_total is None else int(completion_tokens_total)),
        total_tokens=total_tokens,
        token_source=str(token_source),
        token_unavailable_reason=(None if not token_unavailable_reason else str(token_unavailable_reason)),
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_cost_records(
    *,
    method: str,
    dataset: str,
    backend: str,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _collect(name: str) -> List[float]:
        values: List[float] = []
        for row in rows:
            cost = row.get("cost") if isinstance(row, dict) else None
            if not isinstance(cost, dict):
                continue
            value = cost.get(name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _summary(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "sum": None, "avg": None, "p50": None, "p95": None}
        sorted_vals = sorted(values)
        count = len(sorted_vals)
        p50 = sorted_vals[int(0.50 * (count - 1))]
        p95 = sorted_vals[int(0.95 * (count - 1))]
        return {
            "count": count,
            "sum": round(sum(sorted_vals), 3),
            "avg": round(sum(sorted_vals) / count, 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
        }

    prompt_tokens = _collect("prompt_tokens_total")
    completion_tokens = _collect("completion_tokens_total")
    total_tokens = _collect("total_tokens")
    index_times = _collect("index_time_ms")
    query_times = _collect("query_time_ms")
    query_retrieval_times = _collect("query_retrieval_ms")
    query_reader_times = _collect("query_reader_ms")
    llm_calls = _collect("llm_calls")
    llm_retries = _collect("llm_retries")

    missing_cost_count = 0
    token_source_counts: Dict[str, int] = {}
    for row in rows:
        cost = row.get("cost") if isinstance(row, dict) else None
        if not isinstance(cost, dict):
            missing_cost_count += 1
            continue
        src = str(cost.get("token_source") or "unknown")
        token_source_counts[src] = token_source_counts.get(src, 0) + 1

    return {
        "method": method,
        "dataset": dataset,
        "backend": backend,
        "count": len(rows),
        "missing_cost_count": missing_cost_count,
        "token_source_counts": token_source_counts,
        "index_time_ms": _summary(index_times),
        "query_time_ms": _summary(query_times),
        "query_retrieval_ms": _summary(query_retrieval_times),
        "query_reader_ms": _summary(query_reader_times),
        "prompt_tokens_total": _summary(prompt_tokens),
        "completion_tokens_total": _summary(completion_tokens),
        "total_tokens": _summary(total_tokens),
        "llm_calls": _summary(llm_calls),
        "llm_retries": _summary(llm_retries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
