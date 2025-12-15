#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slugify(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _unique_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_doc_slug_from_note_id(note_id: str) -> Optional[str]:
    """
    Note IDs are usually: mirage/<doc_slug>__<query_id>#p0000#0
    Return <doc_slug> if present.
    """
    if not note_id:
        return None
    prefix = "mirage/"
    start = note_id.find(prefix)
    if start < 0:
        return None
    start += len(prefix)
    end = note_id.find("__", start)
    if end < 0:
        return None
    slug = note_id[start:end].strip()
    return slug or None


def _load_mirage_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("MIRAGE dataset must be a JSON list")
    return data


def _gold_doc_slugs(dataset: Sequence[dict]) -> Dict[str, Set[str]]:
    gold: Dict[str, Set[str]] = {}
    for idx, item in enumerate(dataset):
        qid = str(item.get("query_id") or item.get("id") or idx)
        doc_name = item.get("doc_name")
        if not doc_name:
            gold[qid] = set()
            continue
        gold[qid] = {_slugify(str(doc_name))}
    return gold


def _compute_doc_metrics(
    retrieved: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    ks: Sequence[int],
) -> Dict[str, float]:
    ks = [int(k) for k in ks if int(k) > 0]
    if not ks:
        raise ValueError("ks must contain positive integers")
    metrics = {f"DocRecall@{k}": 0.0 for k in ks}
    metrics.update({f"DocPrec@{k}": 0.0 for k in ks})
    metrics.update({f"Hit@{k}": 0.0 for k in ks})
    n = 0
    for qid, gold_set in gold.items():
        if qid not in retrieved:
            continue
        n += 1
        docs = _unique_preserve(retrieved.get(qid) or [])
        for k in ks:
            top_set = set(docs[:k])
            inter = len(top_set & gold_set)
            if gold_set:
                recall = inter / len(gold_set)
            else:
                recall = 1.0 if not top_set else 0.0
            if top_set:
                prec = inter / len(top_set)
            else:
                prec = 1.0 if not gold_set else 0.0
            hit = 1.0 if inter > 0 else (1.0 if not gold_set and not top_set else 0.0)
            metrics[f"DocRecall@{k}"] += recall
            metrics[f"DocPrec@{k}"] += prec
            metrics[f"Hit@{k}"] += hit

    if n == 0:
        return {}
    return {name: value / n for name, value in metrics.items()}


def _render_table(run_name: str, metrics: Dict[str, float], ks: Sequence[int]) -> str:
    ks = [int(k) for k in ks if int(k) > 0]
    headers = ["run"] + [f"DocRecall@{k}" for k in ks] + [f"Hit@{k}" for k in ks]
    header = "| " + " | ".join(headers) + " |\n"
    header += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    row = [run_name]
    for k in ks:
        row.append(f"{metrics.get(f'DocRecall@{k}', 0.0):.3f}")
    for k in ks:
        row.append(f"{metrics.get(f'Hit@{k}', 0.0):.3f}")
    return header + "| " + " | ".join(row) + " |"


@dataclass
class NaiveIndexConfig:
    index_dir: Path
    topk: int


def _eval_naive_index(
    dataset: Sequence[dict],
    *,
    index_dir: Path,
    ks: Sequence[int],
    limit: Optional[int],
    embed_device: str = "auto",
    embed_batch_size: int = 4,
    embed_model: Optional[str] = None,
    embed_max_length: Optional[int] = None,
    embed_normalize: Optional[bool] = None,
) -> Dict[str, float]:
    from baselines.naive_rag.runner import NaiveIndex

    meta_path = index_dir / "meta.json"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index_path = Path(str((meta or {}).get("index") or "")).expanduser()
            chunks_path = Path(str((meta or {}).get("chunks") or "")).expanduser()
        except Exception:
            index_path = index_dir / "index.faiss"
            chunks_path = index_dir / "chunks.jsonl"
    else:
        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.faiss: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl: {chunks_path}")

    max_k = max(int(k) for k in ks)
    meta_model = (meta or {}).get("embed_model") or (meta or {}).get("embedding_model")
    meta_max_len = (meta or {}).get("embed_max_length")
    meta_norm = (meta or {}).get("normalize")
    resolved_model = str(embed_model or meta_model or "Qwen/Qwen3-Embedding-8B")
    resolved_max_len = int(embed_max_length or meta_max_len or 512)
    if embed_normalize is not None:
        resolved_norm = bool(embed_normalize)
    elif meta_norm is not None:
        resolved_norm = bool(meta_norm)
    else:
        resolved_norm = True

    retriever = NaiveIndex(
        str(index_path),
        str(chunks_path),
        embed_model=resolved_model,
        embed_device=str(embed_device),
        embed_batch_size=int(embed_batch_size),
        embed_max_length=resolved_max_len,
        embed_normalize=resolved_norm,
    )

    retrieved: Dict[str, List[str]] = {}
    items = list(dataset)
    if limit and limit > 0:
        items = items[:limit]
    for idx, item in enumerate(items):
        qid = str(item.get("query_id") or item.get("id") or idx)
        question = item.get("query") or item.get("question") or ""
        hits = retriever.retrieve(str(question), k=max_k)
        doc_slugs: List[str] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            doc_title = ((meta.get("meta") or {}).get("doc_title")) or hit.get("title") or ""
            if doc_title:
                doc_slugs.append(_slugify(str(doc_title)))
        retrieved[qid] = doc_slugs

    gold = _gold_doc_slugs(items)
    return _compute_doc_metrics(retrieved, gold, ks)


def _eval_anorag(
    dataset: Sequence[dict],
    *,
    indexes_dir: Path,
    notes_path: Path,
    ks: Sequence[int],
    limit: Optional[int],
    source: str,
    disable_bm25: bool,
    disable_reranker: bool,
) -> Dict[str, float]:
    from config import config as config_loader
    from retriever.note_store import NoteStore
    from retriever.operators import Indexes
    from retriever.pipeline import retrieve_answer

    if source not in {"support_note_ids", "evidence"}:
        raise ValueError("--source must be one of: support_note_ids, evidence")

    cfg = config_loader.load_config()
    cfg.setdefault("retriever", {})
    cfg.setdefault("reranker", {})
    # When evaluating against a specific indexes_dir, ensure embedding/BM25 paths point there,
    # mirroring QueryProcessor's behavior.
    retr_cfg = cfg["retriever"]
    embed_cfg = retr_cfg.setdefault("embedding", {})
    bm25_cfg = retr_cfg.setdefault("bm25", {})
    base_idx = Path(indexes_dir)
    embed_cfg["offline_index_path"] = str(base_idx / "faiss" / "notes.faiss")
    embed_cfg["meta_path"] = str(base_idx / "faiss" / "notes.meta.parquet")
    bm25_cfg["store_path"] = str(base_idx / "bm25" / "notes")

    cfg["reranker"]["enabled"] = False if disable_reranker else bool(cfg["reranker"].get("enabled", False))
    if disable_bm25:
        cfg["retriever"].setdefault("bm25", {})
        cfg["retriever"]["bm25"]["enabled"] = False

    indexes = Indexes(str(indexes_dir))
    note_store = NoteStore(str(notes_path))

    # Reuse a single hybrid retriever instance to avoid re-loading embedding models per query.
    hybrid_inst = None
    try:
        from retriever.hybrid import HybridRetriever

        # Only build if at least one channel is enabled.
        retr_cfg2 = cfg.get("retriever") or {}
        embedding_on = bool((retr_cfg2.get("embedding") or {}).get("enabled"))
        bm25_on = bool((retr_cfg2.get("bm25") or {}).get("enabled"))
        rerank_on = bool((cfg.get("reranker") or {}).get("enabled"))
        if embedding_on or bm25_on or rerank_on:
            hybrid_inst = HybridRetriever(cfg)
    except Exception:
        hybrid_inst = None

    max_k = max(int(k) for k in ks)
    items = list(dataset)
    if limit and limit > 0:
        items = items[:limit]

    retrieved: Dict[str, List[str]] = {}
    for idx, item in enumerate(items):
        qid = str(item.get("query_id") or item.get("id") or idx)
        question = item.get("query") or item.get("question") or ""
        if not question:
            retrieved[qid] = []
            continue
        doc_hint = f"mirage/{qid}" if qid else None
        attr_hint = "occupation" if "occupation" in str(question).lower() else None
        structured = retrieve_answer(
            str(question),
            indexes,
            note_store,
            cfg=cfg,
            hybrid=hybrid_inst,
            doc_hint=doc_hint,
            attribute_hint=attr_hint,
        )
        if source == "evidence":
            note_ids = [str(ev.get("note_id") or "") for ev in (structured.get("evidence") or [])]
        else:
            note_ids = [str(nid) for nid in (structured.get("support_note_ids") or [])]

        doc_slugs: List[str] = []
        for nid in note_ids:
            slug = _extract_doc_slug_from_note_id(nid)
            if slug:
                doc_slugs.append(_slugify(slug))
                continue
            # Fallback: load note and parse meta.source if the id format is unexpected.
            note = note_store.get(nid)
            subj = (note or {}).get("subj")
            if isinstance(subj, str) and subj.strip():
                doc_slugs.append(_slugify(subj))
                continue
            source_val = ((note or {}).get("meta") or {}).get("source")
            if isinstance(source_val, str) and source_val:
                doc_slugs.append(_slugify(source_val.split("/", 1)[-1].split("__", 1)[0]))

        retrieved[qid] = doc_slugs[: max_k * 8]  # keep a few duplicates; metrics will dedup

    gold = _gold_doc_slugs(items)
    return _compute_doc_metrics(retrieved, gold, ks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIRAGE doc-level retrieval recall (Hit/Recall@k).")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="data/mirage_sample/dataset.json", help="Path to MIRAGE dataset.json")
    common.add_argument("--ks", default="1,3,5,10", help="Comma-separated k values (default: 1,3,5,10)")
    common.add_argument("--limit", type=int, default=0, help="Evaluate first N samples (0=all)")
    common.add_argument("--json-out", default=None, help="Optional path to write metrics JSON")

    naive = sub.add_parser("naive", parents=[common], help="Evaluate naive FAISS chunk index recall")
    naive.add_argument("--index-dir", default="result/mirage_naive", help="Directory with index.faiss + chunks.jsonl")
    naive.add_argument(
        "--embed-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Embedding device preference for query embedding (auto prefers CUDA, falls back to CPU)",
    )
    naive.add_argument("--embed-batch-size", type=int, default=4, help="Embedding batch size for query embedding")
    naive.add_argument(
        "--embed-model",
        default=None,
        help="Embedding model name/path for query embedding (default: read from index meta.json)",
    )
    naive.add_argument(
        "--embed-max-length",
        type=int,
        default=0,
        help="Embedding max length for query embedding (default: read from index meta.json)",
    )
    norm = naive.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", help="L2-normalize embeddings")
    norm.add_argument(
        "--no-embed-normalize",
        dest="embed_normalize",
        action="store_false",
        help="Disable L2-normalization",
    )
    naive.set_defaults(embed_normalize=None)

    anorag = sub.add_parser("anorag", parents=[common], help="Evaluate AnoRAG structured/hybrid retrieval recall")
    anorag.add_argument("--indexes-dir", required=True, help="Directory with AnoRAG indexes (entity_to_notes.json etc.)")
    anorag.add_argument("--notes-path", required=True, help="AnoRAG notes.jsonl path")
    anorag.add_argument(
        "--source",
        default="support_note_ids",
        choices=["support_note_ids", "evidence"],
        help="Which structured field to treat as retrieval results (default: support_note_ids)",
    )
    anorag.add_argument("--disable-bm25", action="store_true", help="Disable BM25 during evaluation")
    anorag.add_argument("--disable-reranker", action="store_true", help="Disable LLM reranker during evaluation")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset = _load_mirage_dataset(dataset_path)

    ks = [int(tok) for tok in str(args.ks).split(",") if tok.strip()]
    limit = int(args.limit) if int(args.limit or 0) > 0 else None

    if args.mode == "naive":
        embed_max_length = int(args.embed_max_length) if int(getattr(args, "embed_max_length", 0) or 0) > 0 else None
        metrics = _eval_naive_index(
            dataset,
            index_dir=Path(args.index_dir),
            ks=ks,
            limit=limit,
            embed_device=str(getattr(args, "embed_device", "auto")),
            embed_batch_size=int(getattr(args, "embed_batch_size", 4)),
            embed_model=getattr(args, "embed_model", None),
            embed_max_length=embed_max_length,
            embed_normalize=getattr(args, "embed_normalize", None),
        )
        run_name = Path(args.index_dir).name
    else:
        metrics = _eval_anorag(
            dataset,
            indexes_dir=Path(args.indexes_dir),
            notes_path=Path(args.notes_path),
            ks=ks,
            limit=limit,
            source=str(args.source),
            disable_bm25=bool(getattr(args, "disable_bm25", False)),
            disable_reranker=bool(getattr(args, "disable_reranker", False)),
        )
        run_name = f"anorag:{Path(args.indexes_dir).name}"

    if not metrics:
        print("No metrics computed (no overlap between dataset IDs and retrieved results).")
        return

    print(_render_table(run_name, metrics, ks))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"run": run_name, "metrics": metrics}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
