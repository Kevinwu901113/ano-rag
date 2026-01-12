import argparse
import json
import re
import shutil
import sys
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

from loguru import logger

from relrag.api import build_index, retrieve, answer
from relrag.config.config_loader import config as global_config
from relrag.indexer.bm25_index import BM25IndexBuilder
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from relrag.retriever.note_store import NoteStore
from relrag.utils.output_eval import extract_final_answer, has_final_tag


DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_CACHE_DIR = "result/cache"
DEFAULT_OUTPUT_DIR = "result"
DEFAULT_DEBUG_DIR = "result/debug"
DEFAULT_DEBUG_MAX_NOTES = 50


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _slugify_title(title: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", (title or "")).strip("_").lower()
    cleaned = cleaned or "doc"
    return cleaned[:max_len]


def _make_doc_id(qid: str, idx: int, title: str) -> str:
    slug = _slugify_title(title)
    return f"{qid}_{idx:02d}_{slug}"


def _count_examples(path: Path, limit: int) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            if limit and total >= limit:
                break
    return total


class ProgressBar:
    def __init__(
        self,
        total: int,
        stream: Optional[TextIO] = None,
        min_interval_sec: float = 0.5,
    ) -> None:
        self.total = max(0, int(total))
        self.processed = 0
        self._stream = stream or sys.stderr
        self._min_interval_sec = max(0.05, float(min_interval_sec))
        self._last_update = 0.0
        self._last_len = 0

    def update(self, step: int = 1) -> None:
        self.processed += max(0, int(step))
        now = time.time()
        if self.total and self.processed < self.total:
            if (now - self._last_update) < self._min_interval_sec:
                return
        self._last_update = now
        self._render()

    def _render(self) -> None:
        total = self.total
        processed = min(self.processed, total) if total else self.processed
        remaining = max(0, total - processed) if total else 0
        if total > 0:
            columns = shutil.get_terminal_size((80, 20)).columns
            bar_width = max(10, min(40, columns - 40))
            ratio = min(1.0, processed / total)
            filled = int(bar_width * ratio)
            bar = "=" * filled + "-" * (bar_width - filled)
            msg = f"[{bar}] {processed}/{total} (remaining {remaining})"
        else:
            msg = f"{processed} processed"
        pad = " " * max(0, self._last_len - len(msg))
        self._stream.write("\r" + msg + pad)
        self._stream.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        self._render()
        self._stream.write("\n")
        self._stream.flush()


def _write_docs_for_example(
    example: Dict[str, Any],
    docs_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Dict[str, Any]]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_index: Dict[str, Dict[str, Any]] = {}
    context = example.get("context") or []
    qid = str(example.get("_id") or "unknown")

    for idx, item in enumerate(context):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        doc_id = _make_doc_id(qid, idx, title)
        doc_index[doc_id] = {"title": title, "sentences": sentences}

        text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
        if not text:
            continue
        doc_path = docs_dir / f"{doc_id}.txt"
        if doc_path.exists() and not overwrite:
            continue
        doc_path.write_text(text, encoding="utf-8")

    return doc_index


def _resolve_doc_id_from_source(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    return source.split("#", 1)[0].strip() or None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\W+", "", (text or "").lower())


def _find_sentence_index(evidence: str, sentences: List[str]) -> Optional[int]:
    ev_norm = _normalize_for_match(evidence)
    if not ev_norm:
        return None
    for idx, sent in enumerate(sentences):
        sent_norm = _normalize_for_match(sent)
        if not sent_norm:
            continue
        if sent_norm in ev_norm or ev_norm in sent_norm:
            return idx
    return None


def _build_retrieved_context(
    evidences: List[Dict[str, Any]],
    note_store: NoteStore,
    doc_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for ev in evidences:
        note_id = ev.get("note_id")
        note = note_store.get_weak(note_id) if note_id else None
        source = ((note or {}).get("meta") or {}).get("source") or ev.get("source")
        doc_id = _resolve_doc_id_from_source(source) if source else None
        if not doc_id:
            doc_id = ev.get("doc_id") or _resolve_doc_id_from_source(note_id)
        doc_meta = doc_index.get(doc_id) if doc_id else None
        title = doc_meta.get("title") if doc_meta else None
        sentences = doc_meta.get("sentences") if doc_meta else []
        evidence_text = ev.get("canonical") or ev.get("evidence") or ""
        sentence_idx = _find_sentence_index(evidence_text, sentences) if sentences else None

        contexts.append(
            {
                "note_id": note_id,
                "doc_id": doc_id,
                "title": title,
                "sentence_idx": sentence_idx,
                "evidence": ev.get("evidence"),
                "canonical": ev.get("canonical"),
                "weak": bool(ev.get("weak", False)),
                "score": ev.get("score"),
                "source": source,
            }
        )
    return contexts


def _build_supporting_facts(retrieved_context: List[Dict[str, Any]]) -> List[List[Any]]:
    facts: List[List[Any]] = []
    seen = set()
    for ctx in retrieved_context:
        title = ctx.get("title")
        idx = ctx.get("sentence_idx")
        if title is None or idx is None:
            continue
        key = (title, int(idx))
        if key in seen:
            continue
        facts.append([title, int(idx)])
        seen.add(key)
    return facts


def _collect_debug_notes(
    note_store: NoteStore,
    note_ids: Iterable[str],
    max_notes: int,
) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    limit = max(0, int(max_notes))
    for note_id in note_ids:
        if not note_id or note_id in collected:
            continue
        note = note_store.get_weak(note_id)
        if note:
            collected[note_id] = note
        if limit and len(collected) >= limit:
            break
    return collected


def _write_debug_artifacts(
    record: Dict[str, Any],
    *,
    debug_dir: Path,
    docs_dir: Path,
    notes_path: Path,
    index_dir: Path,
    doc_index: Dict[str, Dict[str, Any]],
    note_store: NoteStore,
    max_notes: int,
) -> Optional[Path]:
    qid = str(record.get("_id") or "unknown")
    debug_dir.mkdir(parents=True, exist_ok=True)
    retrieve_result = (record.get("intermediate") or {}).get("retrieve_result") or {}
    support_note_ids = retrieve_result.get("support_note_ids") or []
    evidence_note_ids = [
        ev.get("note_id")
        for ev in (retrieve_result.get("evidence") or [])
        if ev.get("note_id")
    ]
    merged_note_ids = list(dict.fromkeys(list(support_note_ids) + list(evidence_note_ids)))
    notes = _collect_debug_notes(note_store, merged_note_ids, max_notes=max_notes)

    payload = {
        "qid": qid,
        "question": record.get("question"),
        "paths": {
            "docs_dir": str(docs_dir),
            "notes_path": str(notes_path),
            "indexes_dir": str(index_dir),
        },
        "doc_index": doc_index,
        "note_ids": {
            "support": support_note_ids,
            "evidence": evidence_note_ids,
            "saved": list(notes.keys()),
        },
        "notes": notes,
        "record": record,
    }
    debug_path = debug_dir / f"{qid}.json"
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return debug_path


def _resolve_llm_config(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = global_config.load_config()
    endpoint = args.endpoint or (cfg.get("vllm") or {}).get("endpoint")
    model = args.model or (cfg.get("vllm") or {}).get("model")
    if not endpoint or not model:
        raise ValueError("LLM endpoint/model is required (use args or config)")
    return endpoint, model


def _load_entry_config() -> Dict[str, Any]:
    cfg = global_config.load_config()
    entry_cfg = cfg.get("hotpot_entry") or cfg.get("entry") or {}
    if not isinstance(entry_cfg, dict):
        return {}
    return entry_cfg


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


def _pick_arg(args: argparse.Namespace, entry_cfg: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if name in entry_cfg:
        return entry_cfg.get(name)
    return default


def _ensure_index(
    docs_dir: Path,
    index_root: Path,
    llm_endpoint: str,
    llm_model: str,
    force_build: bool,
) -> Dict[str, Any]:
    notes_path = index_root / "notes.jsonl"
    indexes_dir = index_root / "indexes"
    if not force_build and notes_path.exists() and indexes_dir.exists():
        return {"status": "reused"}
    index_root.mkdir(parents=True, exist_ok=True)
    return build_index(
        docs_input=str(docs_dir),
        output_dir=str(index_root),
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
    )


def _prepare_aux_config(example_root: Path) -> Dict[str, Any]:
    cfg = deepcopy(global_config.load_config())
    notes_path = example_root / "notes.jsonl"
    indexes_root = example_root / "indexes"
    cfg.setdefault("notes", {})["out_path"] = str(notes_path)
    retriever_cfg = cfg.setdefault("retriever", {})
    embed_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    embed_cfg["offline_index_path"] = str(indexes_root / "faiss" / "notes.faiss")
    embed_cfg["meta_path"] = str(indexes_root / "faiss" / "notes.meta.parquet")
    bm25_cfg["store_path"] = str(indexes_root / "bm25" / "notes")
    return cfg


def _build_aux_indexes(example_root: Path) -> Dict[str, Any]:
    cfg = _prepare_aux_config(example_root)
    notes_path = Path(cfg.get("notes", {}).get("out_path", example_root / "notes.jsonl"))
    stats: Dict[str, Any] = {}
    if not notes_path.exists():
        stats["embedding"] = "skipped_missing_notes"
        stats["bm25"] = "skipped_missing_notes"
        return stats

    embed_enabled = bool((cfg.get("retriever") or {}).get("embedding", {}).get("enabled"))
    bm25_enabled = bool((cfg.get("retriever") or {}).get("bm25", {}).get("enabled"))

    if embed_enabled:
        try:
            EmbeddingIndexBuilder(cfg).build()
            stats["embedding"] = "ok"
        except Exception as exc:
            logger.warning("Embedding index build failed {}: {}", notes_path, exc)
            stats["embedding"] = f"error:{exc}"
    else:
        stats["embedding"] = "disabled"

    if bm25_enabled:
        try:
            BM25IndexBuilder(cfg).build()
            stats["bm25"] = "ok"
        except Exception as exc:
            logger.warning("BM25 corpus build failed {}: {}", notes_path, exc)
            stats["bm25"] = f"error:{exc}"
    else:
        stats["bm25"] = "disabled"

    return stats


def _prepare_retriever_config(example_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = _prepare_aux_config(example_root)
    embed_cfg = (cfg.get("retriever") or {}).get("embedding") or {}
    bm25_cfg = (cfg.get("retriever") or {}).get("bm25") or {}

    embed_index = Path(embed_cfg.get("offline_index_path", ""))
    embed_meta = Path(embed_cfg.get("meta_path", ""))
    bm25_corpus = Path(bm25_cfg.get("store_path", "")) / "notes.jsonl"

    embed_ready = embed_index.exists() and embed_meta.exists()
    bm25_ready = bm25_corpus.exists()

    if not embed_ready:
        embed_cfg["enabled"] = False
    if not bm25_ready:
        bm25_cfg["enabled"] = False

    return cfg, {
        "embedding_ready": embed_ready,
        "bm25_ready": bm25_ready,
        "embedding_index": str(embed_index),
        "embedding_meta": str(embed_meta),
        "bm25_corpus": str(bm25_corpus),
    }


def _process_example(
    example: Dict[str, Any],
    cache_root: Path,
    llm_endpoint: str,
    llm_model: str,
    top_k: int,
    force_build: bool,
    debug_dir: Optional[Path],
    debug_max_notes: int,
) -> Dict[str, Any]:
    qid = str(example.get("_id") or "unknown")
    question = str(example.get("question") or "")

    example_root = cache_root / qid
    docs_dir = example_root / "docs"
    doc_index = _write_docs_for_example(example, docs_dir, overwrite=force_build)
    build_stats = _ensure_index(docs_dir, example_root, llm_endpoint, llm_model, force_build)
    aux_stats = _build_aux_indexes(example_root)
    retriever_cfg, retriever_paths = _prepare_retriever_config(example_root)

    notes_path = example_root / "notes.jsonl"
    index_dir = example_root / "indexes"
    retrieve_result = retrieve(
        question=question,
        index_dir=str(index_dir),
        notes_path=str(notes_path),
        top_k=top_k,
        cfg=retriever_cfg,
    )

    evidences = retrieve_result.get("evidence") or []
    raw_answer = answer(
        question=question,
        evidences=evidences,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
    )
    final_answer = extract_final_answer(raw_answer) or raw_answer

    note_store = NoteStore(str(notes_path))
    retrieved_context = _build_retrieved_context(evidences, note_store, doc_index)
    supporting_facts = _build_supporting_facts(retrieved_context)

    output_record = {
        "_id": qid,
        "question": question,
        "answer": final_answer,
        "sp": supporting_facts,
        "generated_answer": final_answer,
        "supporting_facts": supporting_facts,
        "retrieved_context": retrieved_context,
        "intermediate": {
            "build_stats": build_stats,
            "aux_indexes": aux_stats,
            "retriever_paths": retriever_paths,
            "retrieve_result": retrieve_result,
            "structured_answer": retrieve_result.get("answer"),
            "llm_raw": raw_answer,
            "llm_has_final": has_final_tag(raw_answer),
            "support_note_ids": retrieve_result.get("support_note_ids"),
            "paths": retrieve_result.get("paths"),
            "ir": retrieve_result.get("ir"),
            "fallback": retrieve_result.get("fallback"),
            "intent": retrieve_result.get("intent"),
        },
    }
    if debug_dir:
        debug_path = _write_debug_artifacts(
            output_record,
            debug_dir=debug_dir,
            docs_dir=docs_dir,
            notes_path=notes_path,
            index_dir=index_dir,
            doc_index=doc_index,
            note_store=note_store,
            max_notes=debug_max_notes,
        )
        output_record["intermediate"]["debug_path"] = str(debug_path) if debug_path else None

    return output_record


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
    progress: Optional[ProgressBar] = None,
    stall_warn_sec: float = DEFAULT_STALL_WARN_SEC,
    stall_abort_sec: float = DEFAULT_STALL_ABORT_SEC,
) -> Tuple[int, int]:
    completed = 0
    succeeded = 0
    pending = set(future_map)
    last_progress = time.time()
    stall_warn_sec = max(1.0, float(stall_warn_sec))
    stall_abort_sec = float(stall_abort_sec)
    if stall_abort_sec <= 0:
        stall_abort_sec = 0.0
    while pending:
        done, pending = wait(pending, timeout=stall_warn_sec, return_when=FIRST_COMPLETED)
        if not done:
            idle_for = time.time() - last_progress
            sample = [future_map[f] for f in list(pending)[:5]]
            logger.warning(
                "No completed futures for {:.0f}s; still waiting on {} examples (sample: {})",
                idle_for,
                len(pending),
                sample,
            )
            if stall_abort_sec and idle_for >= stall_abort_sec:
                logger.error(
                    "Aborting {} stalled examples after {:.0f}s idle",
                    len(pending),
                    idle_for,
                )
                for future in list(pending):
                    qid = future_map.get(future, "unknown")
                    if not future.cancel():
                        logger.warning("Failed to cancel stalled future qid={}", qid)
                    else:
                        logger.error("Cancelled stalled future qid={}", qid)
                if progress is not None:
                    progress.update(len(pending))
                completed += len(pending)
                pending.clear()
                break
            continue
        for future in done:
            qid = future_map.get(future, "unknown")
            try:
                record = future.result()
            except Exception as exc:
                logger.error("Failed example {}: {}", qid, exc)
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


def _write_official_output(jsonl_path: Path, output_dir: Path, timestamp: int) -> Path:
    answers: Dict[str, Any] = {}
    supports: Dict[str, Any] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record.get("_id") or "")
            if not qid:
                continue
            answers[qid] = record.get("answer") or record.get("generated_answer") or ""
            supports[qid] = record.get("sp") or record.get("supporting_facts") or []
    official_path = output_dir / f"result_{timestamp}_official.json"
    with official_path.open("w", encoding="utf-8") as handle:
        json.dump({"answer": answers, "sp": supports}, handle, ensure_ascii=False)
    return official_path


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA JSONL entry for RelRAG")
    parser.add_argument("--data", help="Path to HotpotQA JSONL dataset (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="LLM model name (defaults to config)")
    parser.add_argument("--top_k", type=int, help="Top-k retrieval fanout (fallback to config)")
    parser.add_argument("--limit", type=int, help="Process only first N examples (fallback to config)")
    parser.add_argument("--workers", type=int, help="Parallel workers (single process, fallback to config)")
    parser.add_argument("--cache_dir", help="Cache root for per-question indexes (fallback to config)")
    parser.add_argument("--output_dir", help="Output directory (fallback to config)")
    parser.add_argument("--force_build", action="store_true", help="Rebuild indexes even if cached")
    parser.add_argument("--debug_dir", help="Debug artifacts output directory (set empty to disable, fallback to config)")
    parser.add_argument("--debug_max_notes", type=int, help="Max notes to store per question in debug dump (fallback to config)")
    parser.add_argument("--stall_warn_sec", type=float, help="Warn if no worker finishes within this many seconds (fallback to config)")
    parser.add_argument("--stall_abort_sec", type=float, help="Abort pending workers after this many idle seconds (0 to disable, fallback to config)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else repo_root / path

    entry_cfg = _load_entry_config()
    args.data = _pick_arg(args, entry_cfg, "data", None)
    if not args.data:
        raise ValueError("Dataset path missing. Provide --data or set hotpot_entry.data in config.")
    args.cache_dir = _pick_arg(args, entry_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _coerce_int(_pick_arg(args, entry_cfg, "top_k", DEFAULT_TOP_K), DEFAULT_TOP_K)
    args.limit = _coerce_int(_pick_arg(args, entry_cfg, "limit", DEFAULT_LIMIT), DEFAULT_LIMIT)
    args.workers = _coerce_int(_pick_arg(args, entry_cfg, "workers", DEFAULT_WORKERS), DEFAULT_WORKERS)
    args.debug_dir = _pick_arg(args, entry_cfg, "debug_dir", DEFAULT_DEBUG_DIR)
    args.debug_max_notes = _coerce_int(
        _pick_arg(args, entry_cfg, "debug_max_notes", DEFAULT_DEBUG_MAX_NOTES),
        DEFAULT_DEBUG_MAX_NOTES,
    )
    args.stall_warn_sec = _coerce_float(
        _pick_arg(args, entry_cfg, "stall_warn_sec", DEFAULT_STALL_WARN_SEC),
        DEFAULT_STALL_WARN_SEC,
    )
    args.stall_abort_sec = _coerce_float(
        _pick_arg(args, entry_cfg, "stall_abort_sec", DEFAULT_STALL_ABORT_SEC),
        DEFAULT_STALL_ABORT_SEC,
    )
    if entry_cfg.get("force_build"):
        args.force_build = True

    data_path = _resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    llm_endpoint, llm_model = _resolve_llm_config(args)
    cache_root = _resolve_path(args.cache_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"result_{timestamp}.jsonl"
    total_examples = _count_examples(data_path, args.limit)
    progress = ProgressBar(total_examples)
    debug_dir = _resolve_path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing results to {}", output_path)
    processed = 0
    completed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if args.workers <= 1:
            for example in _load_jsonl(data_path):
                if args.limit and completed >= args.limit:
                    break
                try:
                    record = _process_example(
                        example,
                        cache_root=cache_root,
                        llm_endpoint=llm_endpoint,
                        llm_model=llm_model,
                        top_k=args.top_k,
                        force_build=args.force_build,
                        debug_dir=debug_dir,
                        debug_max_notes=args.debug_max_notes,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    processed += 1
                except Exception as exc:
                    qid = example.get("_id")
                    logger.error("Failed example {}: {}", qid, exc)
                finally:
                    completed += 1
                    progress.update(1)
        else:
            max_workers = max(1, int(args.workers))
            future_map: Dict[Any, str] = {}
            scheduled = 0
            buffer_cap = max_workers * 2
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for example in _load_jsonl(data_path):
                    if args.limit and scheduled >= args.limit:
                        break
                    qid = str(example.get("_id") or "unknown")
                    future = executor.submit(
                        _process_example,
                        example,
                        cache_root,
                        llm_endpoint,
                        llm_model,
                        args.top_k,
                        args.force_build,
                        debug_dir,
                        args.debug_max_notes,
                    )
                    future_map[future] = qid
                    scheduled += 1
                    if len(future_map) >= buffer_cap:
                        done_count, ok_count = _drain_futures(
                            future_map,
                            handle,
                            progress=progress,
                            stall_warn_sec=args.stall_warn_sec,
                            stall_abort_sec=args.stall_abort_sec,
                        )
                        completed += done_count
                        processed += ok_count
                        future_map = {}
                if future_map:
                    done_count, ok_count = _drain_futures(
                        future_map,
                        handle,
                        progress=progress,
                        stall_warn_sec=args.stall_warn_sec,
                        stall_abort_sec=args.stall_abort_sec,
                    )
                    completed += done_count
                    processed += ok_count

    progress.close()
    failed = completed - processed
    logger.info("Completed {} examples (failed {})", processed, failed)
    official_path = _write_official_output(output_path, output_dir, timestamp)
    logger.info("Official-format output written to {}", official_path)


if __name__ == "__main__":
    main()
