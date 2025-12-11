import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from utils.answer_cleaner import _strip_reasoning, clean_model_answer

# Cache for the closure (get_embedding_model)
_embedding_lock = threading.Lock()
_embedding_model_cache: Dict[Tuple[str, str, str], Callable[[List[str]], np.ndarray]] = {}

# Cache for the actual loaded models (tokenizer, model)
_transformers_lock = threading.Lock()
_transformers_cache: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}
_transformers_inference_semaphores: Dict[Tuple[str, str, str], threading.Semaphore] = {}


def _get_inference_semaphore(key: Tuple[str, str, str]) -> threading.Semaphore:
    """
    Return a semaphore to control concurrent embedding calls for a given (model, device).
    Default concurrency can be tuned via EMB_MAX_CONCURRENCY env var (default 4).
    """
    if key in _transformers_inference_semaphores:
        return _transformers_inference_semaphores[key]

    with _transformers_lock:
        if key in _transformers_inference_semaphores:
            return _transformers_inference_semaphores[key]
        try:
            max_conc = int(os.environ.get("EMB_MAX_CONCURRENCY", "4"))
        except ValueError:
            max_conc = 4
        max_conc = max(1, max_conc)
        _transformers_inference_semaphores[key] = threading.Semaphore(max_conc)
        return _transformers_inference_semaphores[key]


def _resolve_torch_dtype(device: str, torch_dtype: Optional[str]) -> Optional[str]:
    """
    Decide which dtype to use. Default to float16 on CUDA to save memory unless overridden.
    """
    if torch_dtype:
        return torch_dtype
    if str(device).startswith("cuda"):
        return "float16"
    return None


def get_embedding_model(
    model_name: str, device: str = "cpu", torch_dtype: Optional[str] = None
) -> Callable[[List[str]], np.ndarray]:
    """
    Returns a callable encoder.
    Ensures the closure is created once per (model_name, device).
    The underlying model loading is also cached.
    """
    default_dtype = _resolve_torch_dtype(device, torch_dtype)
    dtype_key = default_dtype or "auto"
    key = (model_name, device, dtype_key)
    if key in _embedding_model_cache:
        return _embedding_model_cache[key]

    with _embedding_lock:
        if key in _embedding_model_cache:
            return _embedding_model_cache[key]

        def _encoder(texts: List[str], *, torch_dtype: Optional[str] = None) -> np.ndarray:
            dtype_to_use = torch_dtype or default_dtype
            return encode_passages(
                texts,
                model_name=model_name,
                device=device,
                torch_dtype=dtype_to_use,
            )

        _embedding_model_cache[key] = _encoder
        return _encoder


def _load_transformers_model(model_name: str, device: str, torch_dtype: Optional[str] = None):
    dtype_key = torch_dtype or "auto"
    key = (model_name, device, dtype_key)
    if key in _transformers_cache:
        return _transformers_cache[key]

    with _transformers_lock:
        if key in _transformers_cache:
            return _transformers_cache[key]

        print(f"Loading embedding model: {model_name} on {device} (dtype={dtype_key})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = None
        if torch_dtype:
            try:
                dtype = getattr(torch, torch_dtype)
            except AttributeError:
                dtype = None
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

        target_device = torch.device(device)
        model.to(target_device)
        model.eval()

        _transformers_cache[key] = (tokenizer, model)
        return tokenizer, model


def encode_passages(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    device: str = "cpu",
    batch_size: int = 32,
    normalize: bool = True,
    torch_dtype: Optional[str] = None,
) -> np.ndarray:
    """
    Unified embedding interface using transformers.
    Defaults to float16 on CUDA unless torch_dtype is explicitly provided.
    """
    if not texts:
        return np.array([])

    resolved_dtype = _resolve_torch_dtype(device, torch_dtype)
    dtype_key = resolved_dtype or "auto"

    key = (model_name, device, dtype_key)
    tokenizer, model = _load_transformers_model(model_name, device, torch_dtype=resolved_dtype)
    infer_gate = _get_inference_semaphore(key)

    all_embeddings = []

    # Batch processing (bounded concurrency per model/device)
    with infer_gate:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # Default max length
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    return np.array([])


def build_passages_from_context(context: Any, max_passages: int = 10) -> List[str]:
    """
    Normalize HotpotQA context into a list of passages.
    Supports both dict format {"title": [...], "sentences": [...]} and
    list format [[title, [sent1, ...]], ...].
    """
    return [entry["text"] for entry in build_passage_entries(context, max_passages=max_passages)]


def build_passage_entries(context: Any, max_passages: int = 10) -> List[Dict[str, Any]]:
    """
    Normalize HotpotQA context and keep metadata for retrieval logging.
    Each entry has: text, doc_id (title), sent_ids (indices of included sentences).
    """
    entries: List[Dict[str, Any]] = []

    if isinstance(context, dict):
        titles = list(context.get("title", []))[:max_passages]
        sentences_list = list(context.get("sentences", []))[:max_passages]
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            entries.append(
                {
                    "text": f"Title: {title}\nContent: {text}",
                    "doc_id": title,
                    "sent_ids": list(range(len(sentences))),
                    "passage_id": None,
                }
            )
    else:
        for title, sentences in list(context)[:max_passages]:
            text = " ".join(sentences)
            entries.append(
                {
                    "text": f"Title: {title}\nContent: {text}",
                    "doc_id": title,
                    "sent_ids": list(range(len(sentences))),
                    "passage_id": None,
                }
            )

    return entries


def format_context(passages: List[str]) -> str:
    """Join passages with spacing for prompting."""
    return "\n\n".join(passages)


def clean_hotpot_answer(text: str) -> str:
    """
    Strip <think>...</think> and common prefixes to keep only the final short answer.
    Provides fallback to avoid empty outputs.
    """
    if text is None:
        return ""
    raw = str(text)
    cleaned = clean_model_answer(raw)
    cleaned = _strip_reasoning(cleaned)
    cleaned = re.sub(r"^answer\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("\"'“”‘’").strip()
    cleaned = " ".join(cleaned.split())
    if cleaned:
        return cleaned
    fallback = re.sub(r"^answer\s*[:：]\s*", "", raw, flags=re.IGNORECASE).strip()
    fallback = fallback.strip("\"'“”‘’").strip()
    fallback = " ".join(fallback.split())
    return fallback or "Insufficient evidence"

def detect_device(prefer: str = "cuda") -> str:
    """
    Choose device for embedding; default prefers CUDA if available.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    """
    Create or reuse a workspace under root with the given prefix.
    Mirrors the MIRAGE runner behaviour.
    """
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        target = root / f"{prefix}_{len(existing):03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]


def save_predictions_and_qa(
    work_dir: Path,
    predictions: Dict[str, Dict[str, Any]],
    qa_rows: List[Tuple[str, str]],
    *,
    output_path: Optional[Path] = None,
    qa_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Save HotpotQA prediction json (answer/sp dict) and a QA tsv log.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else work_dir / "pred.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    qa_file = Path(qa_path) if qa_path else work_dir / "qa.tsv"
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    qa_lines = []
    for question, answer in qa_rows:
        q = " ".join((question or "").replace("\t", " ").split())
        a = " ".join((answer or "").replace("\t", " ").split())
        qa_lines.append(f"{q}\t{a}")
    qa_file.write_text("\n".join(qa_lines), encoding="utf-8")
    return out_path, qa_file


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    """
    Truncate text to at most max_chars (if provided and positive).
    """
    if max_chars is None or max_chars <= 0:
        return text
    if text and len(text) > max_chars:
        return text[:max_chars]
    return text
