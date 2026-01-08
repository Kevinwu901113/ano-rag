import argparse
import json
import re
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from relrag.api import build_index, retrieve, answer
from relrag.config.config_loader import config as global_config
from relrag.indexer.bm25_index import BM25IndexBuilder
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from relrag.retriever.note_store import NoteStore
from relrag.utils.output_eval import extract_final_answer, has_final_tag


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
        source = ((note or {}).get("meta") or {}).get("source")
        doc_id = _resolve_doc_id_from_source(source)
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


def _resolve_llm_config(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = global_config.load_config()
    endpoint = args.endpoint or (cfg.get("vllm") or {}).get("endpoint")
    model = args.model or (cfg.get("vllm") or {}).get("model")
    if not endpoint or not model:
        raise ValueError("LLM endpoint/model is required (use args or config)")
    return endpoint, model


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

    return output_record


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
) -> int:
    processed = 0
    for future in as_completed(future_map):
        qid = future_map[future]
        try:
            record = future.result()
        except Exception as exc:
            logger.error("Failed example {}: {}", qid, exc)
            continue
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        processed += 1
    return processed


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
    parser.add_argument("--data", required=True, help="Path to HotpotQA JSONL dataset")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="LLM model name (defaults to config)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k retrieval fanout")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N examples")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (single process)")
    parser.add_argument("--cache_dir", default="result/cache", help="Cache root for per-question indexes")
    parser.add_argument("--output_dir", default="result", help="Output directory")
    parser.add_argument("--force_build", action="store_true", help="Rebuild indexes even if cached")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else repo_root / path

    data_path = _resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    llm_endpoint, llm_model = _resolve_llm_config(args)
    cache_root = _resolve_path(args.cache_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"result_{timestamp}.jsonl"

    logger.info("Writing results to {}", output_path)
    processed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if args.workers <= 1:
            for example in _load_jsonl(data_path):
                if args.limit and processed >= args.limit:
                    break
                try:
                    record = _process_example(
                        example,
                        cache_root=cache_root,
                        llm_endpoint=llm_endpoint,
                        llm_model=llm_model,
                        top_k=args.top_k,
                        force_build=args.force_build,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    processed += 1
                except Exception as exc:
                    qid = example.get("_id")
                    logger.error("Failed example {}: {}", qid, exc)
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
                    )
                    future_map[future] = qid
                    scheduled += 1
                    if len(future_map) >= buffer_cap:
                        processed += _drain_futures(future_map, handle)
                        future_map = {}
                if future_map:
                    processed += _drain_futures(future_map, handle)

    logger.info("Completed {} examples", processed)
    official_path = _write_official_output(output_path, output_dir, timestamp)
    logger.info("Official-format output written to {}", official_path)


if __name__ == "__main__":
    main()
