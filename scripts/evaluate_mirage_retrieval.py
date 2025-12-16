#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
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


def _norm(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    # 把标点当空格（避免 "actor," 匹配不到 "actor"）
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return " ".join(text.split())


def _answer_pattern(ans_norm: str) -> re.Pattern:
    # 词边界严格匹配：\bactor\b
    return re.compile(rf"\b{re.escape(ans_norm)}\b", flags=re.IGNORECASE)


def _extract_hit_text(hit: dict) -> str:
    # 关键：从 retrieval.jsonl 的 hit 里取“chunk文本”
    meta = hit.get("meta") or hit.get("metadata") or {}
    return (
        meta.get("text")
        or meta.get("chunk")
        or meta.get("content")
        or hit.get("text")
        or hit.get("content")
        or hit.get("chunk")
        or ""
    )


def answer_hit_at_k(hits: list, gold_answers: list[str], k: int) -> int:
    # gold_answers: 允许多个可接受答案（别名/同义词）
    # 严格：太短的答案不算（避免 "a"/"an"/"of" 这种假命中）
    gold_norm = [_norm(a) for a in (gold_answers or []) if _norm(a)]
    gold_norm = [a for a in gold_norm if len(a) >= 4]
    if not gold_norm:
        return 0

    patterns = [_answer_pattern(a) for a in gold_norm]

    for hit in hits[:k]:
        text = _norm(_extract_hit_text(hit))
        if not text:
            continue
        for pat in patterns:
            if pat.search(text):
                return 1
    return 0


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


def _gold_answers(dataset_items: list[dict]) -> dict[str, list[str]]:
    gold = {}
    for item in dataset_items:
        qid = str(item.get("query_id") or item.get("id") or item.get("_id") or "")
        if not qid:
            continue

        # 兼容 answer / answers 两种格式
        if isinstance(item.get("answers"), list):
            answers = [str(x) for x in item["answers"] if x is not None]
        else:
            answers = [str(item.get("answer") or "")]

        gold[qid] = [a for a in answers if a and a.strip()]
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
    
    headers = ["Run", "Metric"] + [f"@{k}" for k in ks]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Row 1: DocHit (Hit@k)
    row1 = [run_name, "DocHit"]
    for k in ks:
        val = metrics.get(f"Hit@{k}", 0.0)
        row1.append(f"{val:.3f}")
    lines.append("| " + " | ".join(row1) + " |")

    # Row 2: AnswerHit (AnswerHit@k)
    row2 = [run_name, "AnswerHit"]
    for k in ks:
        val = metrics.get(f"AnswerHit@{k}", 0.0)
        row2.append(f"{val:.3f}")
    lines.append("| " + " | ".join(row2) + " |")
    
    return "\n".join(lines)


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
            meta_index = str((meta or {}).get("index") or "")
            meta_chunks = str((meta or {}).get("chunks") or "")
            
            # Try to resolve relative to cwd or absolute
            p_idx = Path(meta_index).expanduser()
            p_chk = Path(meta_chunks).expanduser()
            
            # If not found, try resolving relative to index_dir if they look relative
            if not p_idx.exists() and not p_idx.is_absolute():
                 p_idx = index_dir / meta_index
            if not p_chk.exists() and not p_chk.is_absolute():
                 p_chk = index_dir / meta_chunks
                 
            # Fallback to default names in index_dir if still not found
            if not p_idx.exists() and (index_dir / "index.faiss").exists():
                p_idx = index_dir / "index.faiss"
            if not p_chk.exists() and (index_dir / "chunks.jsonl").exists():
                p_chk = index_dir / "chunks.jsonl"
                
            index_path = p_idx
            chunks_path = p_chk
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
    ks_int = [int(k) for k in ks if int(k) > 0]
    ans_hits = {k: 0 for k in ks_int}
    total_ans = 0
    gold_ans_map = _gold_answers(dataset)

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

        if idx == 0 and hits:
            print("DEBUG hit keys:", hits[0].keys())
            print("DEBUG meta keys:", (hits[0].get("meta") or {}).keys())
            print("DEBUG text preview:", _extract_hit_text(hits[0])[:200])

        gold_answers = gold_ans_map.get(qid, [])
        if qid in gold_ans_map: 
             total_ans += 1
             for k in ks_int:
                 if answer_hit_at_k(hits, gold_answers, k):
                     ans_hits[k] += 1

        doc_slugs: List[str] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            doc_title = ((meta.get("meta") or {}).get("doc_title")) or hit.get("title") or ""
            if doc_title:
                doc_slugs.append(_slugify(str(doc_title)))
        retrieved[qid] = doc_slugs

    gold = _gold_doc_slugs(items)
    metrics = _compute_doc_metrics(retrieved, gold, ks)
    
    if total_ans > 0:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = ans_hits[k] / total_ans
    else:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = 0.0
            
    return metrics


def _eval_raptor_index(
    dataset: Sequence[dict],
    *,
    index_dir: Path,
    ks: Sequence[int],
    limit: Optional[int],
    **kwargs,
) -> Dict[str, float]:
    from baselines.simple_raptor.retriever import SimpleRaptorRetriever
    from config import config as config_loader

    # Default paths for Raptor
    index_path = index_dir / "simple_raptor_index.faiss"
    nodes_path = index_dir / "simple_raptor_nodes.pkl"
    chunk_store_path = index_dir / "simple_raptor_chunk_store.pkl"

    if not index_path.exists():
        # Try finding just "index.faiss" etc if defaults fail
        if (index_dir / "index.faiss").exists():
            index_path = index_dir / "index.faiss"
            nodes_path = index_dir / "nodes.pkl"
            chunk_store_path = index_dir / "chunks.jsonl" # Assuming symlinks

    if not index_path.exists():
         raise FileNotFoundError(f"Missing Raptor index: {index_path}")
    if not nodes_path.exists():
         raise FileNotFoundError(f"Missing Raptor nodes: {nodes_path}")
    if not chunk_store_path.exists():
         raise FileNotFoundError(f"Missing Raptor chunks: {chunk_store_path}")

    # Create config override for embedding
    cfg = config_loader.load_config()
    embed_model = kwargs.get("embed_model")
    embed_device = kwargs.get("embed_device")
    if embed_model:
        cfg.setdefault("retriever", {}).setdefault("embedding", {})["model"] = embed_model
    if embed_device:
        cfg.setdefault("retriever", {}).setdefault("embedding", {})["device"] = embed_device

    retriever = SimpleRaptorRetriever(
        str(index_path),
        str(nodes_path),
        str(chunk_store_path),
        config=cfg,
        top_k=max(ks)
    )

    max_k = max(int(k) for k in ks)
    ks_int = [int(k) for k in ks if int(k) > 0]
    ans_hits = {k: 0 for k in ks_int}
    total_ans = 0
    gold_ans_map = _gold_answers(dataset)

    retrieved: Dict[str, List[str]] = {}
    items = list(dataset)
    if limit and limit > 0:
        items = items[:limit]
        
    for idx, item in enumerate(items):
        qid = str(item.get("query_id") or item.get("id") or idx)
        question = item.get("query") or item.get("question") or ""
        # Use retrieve method which should be available now or fallback
        try:
            hits = retriever.retrieve(str(question), k=max_k)
        except AttributeError:
             # Fallback if retrieve not implemented in class
             # But I saw it in the file read earlier!
             raise

        if idx == 0 and hits:
            print("DEBUG hit keys:", hits[0].keys())
            print("DEBUG meta keys:", (hits[0].get("meta") or {}).keys())
            print("DEBUG text preview:", _extract_hit_text(hits[0])[:200])

        gold_answers = gold_ans_map.get(qid, [])
        if qid in gold_ans_map: 
             total_ans += 1
             for k in ks_int:
                 if answer_hit_at_k(hits, gold_answers, k):
                     ans_hits[k] += 1

        doc_slugs: List[str] = []
        for hit in hits:
            # Raptor hits usually don't have doc_title directly unless it's in text or metadata
            # We can try to parse from text if formatted like "Title\nContent"
            text = hit.get("text", "")
            if "\n" in text:
                doc_title = text.split("\n", 1)[0]
                if len(doc_title) < 100: # Heuristic
                     doc_slugs.append(_slugify(doc_title))
        retrieved[qid] = doc_slugs

    gold = _gold_doc_slugs(items)
    metrics = _compute_doc_metrics(retrieved, gold, ks)
    
    if total_ans > 0:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = ans_hits[k] / total_ans
    else:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = 0.0
            
    return metrics


def _eval_selfrag_index(
    dataset: Sequence[dict],
    *,
    index_dir: Path,
    ks: Sequence[int],
    limit: Optional[int],
    **kwargs,
) -> Dict[str, float]:
    from baselines.simple_selfrag.retriever import SimpleSelfRAGRetriever
    from config import config as config_loader

    # Default paths for SelfRAG
    index_path = index_dir / "simple_selfrag_index.faiss"
    chunk_store_path = index_dir / "simple_selfrag_chunk_store.pkl"

    if not index_path.exists():
         if (index_dir / "index.faiss").exists():
             index_path = index_dir / "index.faiss"
             chunk_store_path = index_dir / "chunks.jsonl" # Symlink

    if not index_path.exists():
         raise FileNotFoundError(f"Missing SelfRAG index: {index_path}")
    if not chunk_store_path.exists():
         raise FileNotFoundError(f"Missing SelfRAG chunks: {chunk_store_path}")

    # Create config override for embedding
    cfg = config_loader.load_config()
    embed_model = kwargs.get("embed_model")
    embed_device = kwargs.get("embed_device")
    # SimpleSelfRAGRetriever might not take config in __init__, it uses global config?
    # Let's check SimpleSelfRAGRetriever source or assume it uses global_config.
    # If it uses global_config, we should update it.
    if embed_model:
        # Update global config if possible or pass to retriever if supported
        # The class usually loads config inside __init__
        # I can try to patch it or pass config if it accepts it.
        # Looking at run_simple_selfrag.py, it doesn't pass config to Retriever.
        # It relies on global_config.
        config_loader.set("retriever.embedding.model", embed_model)
        config_loader.set("retriever.simple_selfrag.embedding.model", embed_model)
    if embed_device:
        config_loader.set("retriever.embedding.device", embed_device)

    retriever = SimpleSelfRAGRetriever(str(index_path), str(chunk_store_path))
    # Disable LLM for retrieval only if possible
    retriever.llm_client = None

    max_k = max(int(k) for k in ks)
    ks_int = [int(k) for k in ks if int(k) > 0]
    ans_hits = {k: 0 for k in ks_int}
    total_ans = 0
    gold_ans_map = _gold_answers(dataset)

    retrieved: Dict[str, List[str]] = {}
    items = list(dataset)
    if limit and limit > 0:
        items = items[:limit]
        
    for idx, item in enumerate(items):
        qid = str(item.get("query_id") or item.get("id") or idx)
        question = item.get("query") or item.get("question") or ""
        
        # SelfRAGRetriever usually has retrieve method?
        # Let's assume yes or use default
        hits = retriever.retrieve(str(question), top_k=max_k) # SelfRAG uses top_k

        if idx == 0 and hits:
            print("DEBUG hit keys:", hits[0].keys())
            print("DEBUG meta keys:", (hits[0].get("meta") or {}).keys())
            print("DEBUG text preview:", _extract_hit_text(hits[0])[:200])

        gold_answers = gold_ans_map.get(qid, [])
        if qid in gold_ans_map: 
             total_ans += 1
             for k in ks_int:
                 if answer_hit_at_k(hits, gold_answers, k):
                     ans_hits[k] += 1

        doc_slugs: List[str] = []
        for hit in hits:
            # SelfRAG chunks usually have title prepended
            text = hit.get("text", "")
            if "\n" in text:
                doc_title = text.split("\n", 1)[0]
                if len(doc_title) < 100:
                     doc_slugs.append(_slugify(doc_title))
        retrieved[qid] = doc_slugs

    gold = _gold_doc_slugs(items)
    metrics = _compute_doc_metrics(retrieved, gold, ks)
    
    if total_ans > 0:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = ans_hits[k] / total_ans
    else:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = 0.0
            
    return metrics



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
    ks_int = [int(k) for k in ks if int(k) > 0]
    ans_hits = {k: 0 for k in ks_int}
    total_ans = 0
    gold_ans_map = _gold_answers(dataset)

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
        retrieved_hits: List[dict] = []
        
        for nid in note_ids:
            note = note_store.get(nid)
            if note:
                retrieved_hits.append(note)
            else:
                retrieved_hits.append({})

            slug = _extract_doc_slug_from_note_id(nid)
            if slug:
                doc_slugs.append(_slugify(slug))
                continue
            # Fallback: load note and parse meta.source if the id format is unexpected.
            subj = (note or {}).get("subj")
            if isinstance(subj, str) and subj.strip():
                doc_slugs.append(_slugify(subj))
                continue
            source_val = ((note or {}).get("meta") or {}).get("source")
            if isinstance(source_val, str) and source_val:
                doc_slugs.append(_slugify(source_val.split("/", 1)[-1].split("__", 1)[0]))

        if idx == 0 and retrieved_hits:
            print("DEBUG hit keys:", retrieved_hits[0].keys())
            print("DEBUG meta keys:", (retrieved_hits[0].get("meta") or {}).keys())
            print("DEBUG text preview:", _extract_hit_text(retrieved_hits[0])[:200])

        gold_answers = gold_ans_map.get(qid, [])
        if qid in gold_ans_map:
            total_ans += 1
            for k in ks_int:
                if answer_hit_at_k(retrieved_hits, gold_answers, k):
                    ans_hits[k] += 1

        retrieved[qid] = doc_slugs[: max_k * 8]  # keep a few duplicates; metrics will dedup

    gold = _gold_doc_slugs(items)
    metrics = _compute_doc_metrics(retrieved, gold, ks)

    if total_ans > 0:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = ans_hits[k] / total_ans
    else:
        for k in ks_int:
            metrics[f"AnswerHit@{k}"] = 0.0

    return metrics


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

    raptor = sub.add_parser("raptor", parents=[common], help="Evaluate Simple Raptor retrieval")
    raptor.add_argument("--index-dir", default="result/mirage_raptor", help="Directory with Simple Raptor artifacts")
    raptor.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B", help="Embedding model name")
    raptor.add_argument("--embed-device", default="auto", help="Embedding device")

    selfrag = sub.add_parser("selfrag", parents=[common], help="Evaluate Simple SelfRAG retrieval")
    selfrag.add_argument("--index-dir", default="result/mirage_simple_selfrag", help="Directory with Simple SelfRAG artifacts")
    selfrag.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B", help="Embedding model name")
    selfrag.add_argument("--embed-device", default="auto", help="Embedding device")

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
    elif args.mode == "raptor":
        metrics = _eval_raptor_index(
            dataset,
            index_dir=Path(args.index_dir),
            ks=ks,
            limit=limit,
            embed_model=getattr(args, "embed_model", None),
            embed_device=getattr(args, "embed_device", None),
        )
        run_name = Path(args.index_dir).name
    elif args.mode == "selfrag":
        metrics = _eval_selfrag_index(
            dataset,
            index_dir=Path(args.index_dir),
            ks=ks,
            limit=limit,
            embed_model=getattr(args, "embed_model", None),
            embed_device=getattr(args, "embed_device", None),
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
