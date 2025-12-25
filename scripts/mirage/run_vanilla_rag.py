import argparse
import hashlib
import json
import os
import pickle
import re
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now we can import from project modules
try:
    from config.config_loader import config as global_config
except ImportError:
    logger.warning("Could not import config.config_loader; ensure PYTHONPATH is set correctly.")
    pass

from baselines.common.model_clients import get_default_llm_client
from baselines.vanilla_rag import get_retriever
from baselines.vanilla_rag.index import VanillaRAGIndexer
from utils.bm25 import BM25Index, rrf_fuse
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved

_CACHE_KEY_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _slugify(text: str) -> str:
    return _CACHE_KEY_SAFE.sub("_", text or "").strip("_") or "unknown"


def _infer_dataset_id(dataset_path: Path) -> str:
    stem = dataset_path.stem
    parent = dataset_path.parent.name
    if stem in {"dataset", "data"}:
        return parent or stem
    if stem.startswith("dataset_") and parent:
        return f"{parent}_{stem[len('dataset_'):]}"
    return stem


def _resolve_embed_model(embed_cfg: Dict[str, Any]) -> str:
    override = embed_cfg.get("model_path_override")
    base = embed_cfg.get("model", DEFAULT_EMBED_MODEL)
    candidate = str(override or base).strip()
    return candidate or DEFAULT_EMBED_MODEL


def _resolve_embed_device(embed_cfg: Dict[str, Any], cfg_snapshot: Dict[str, Any]) -> str:
    device = embed_cfg.get("device")
    if device:
        return str(device)
    system_cfg = cfg_snapshot.get("system") or {}
    return str(system_cfg.get("device") or "cpu")


def _build_cache_key(
    dataset_id: str,
    doc_pool_hash: str,
    embed_cfg: Dict[str, Any],
    cfg_snapshot: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    chunk_cfg = {"target_tokens": 512, "max_tokens": 600, "overlap_tokens": 50}
    model = _resolve_embed_model(embed_cfg)
    device = _resolve_embed_device(embed_cfg, cfg_snapshot)
    max_len = int(embed_cfg.get("max_len_note", 512))
    normalize = bool(embed_cfg.get("normalize", True))
    dtype = embed_cfg.get("dtype") or "auto"
    provider = embed_cfg.get("provider", "qwen3")
    faiss_kind = "FlatIP"

    key = "__".join(
        [
            _slugify(dataset_id),
            _slugify(model),
            _slugify(device),
            f"len{max_len}",
            f"norm{1 if normalize else 0}",
            f"dtype{_slugify(str(dtype))}",
            f"prov{_slugify(str(provider))}",
            f"chunk{chunk_cfg['target_tokens']}x{chunk_cfg['max_tokens']}x{chunk_cfg['overlap_tokens']}",
            f"faiss{faiss_kind}",
            f"doc{doc_pool_hash[:12]}",
        ]
    )
    meta = {
        "dataset_id": dataset_id,
        "doc_pool_hash": doc_pool_hash,
        "embed_model": model,
        "embed_device": device,
        "embed_max_length": max_len,
        "embed_normalize": normalize,
        "embed_dtype": dtype,
        "embed_provider": provider,
        "chunking": chunk_cfg,
        "faiss_kind": faiss_kind,
    }
    return key, meta


def _load_doc_pool(doc_pool_path: Path) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    with open(doc_pool_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                base_id = item.get("doc_id") or item.get("mapped_id") or str(i)
                doc_id = f"{base_id}::{i}"
                text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
                title = item.get("doc_name", "")
                if title:
                    text = f"{title}\n{text}"
                docs[doc_id] = text
        elif isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, str):
                    docs[k] = v
                elif isinstance(v, dict):
                    docs[k] = v.get("text") or v.get("content") or ""
    return docs

def _select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        next_idx = len(existing)
        target = root / f"{prefix}_{next_idx:03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on MIRAGE")
    parser.add_argument("--dataset-path", type=str, default="data/mirage/mirage_dataset.json", help="Path to MIRAGE dataset")
    parser.add_argument("--result-root", type=str, default="result_relrag", help="Root directory for results")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", type=str, help="Specific working directory (optional)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--lm-endpoint", type=str, help="LLM endpoint override")
    parser.add_argument("--lm-model", type=str, help="LLM model name override")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for LLM decoding")
    parser.add_argument("--index-path", type=str, help="Path to FAISS index (optional, default to artifacts/vanilla_rag_index.faiss)")
    parser.add_argument("--chunk-store-path", type=str, help="Path to chunk store (optional, default to artifacts/vanilla_rag_chunk_store.pkl)")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--topk", type=int, default=10, help="Number of passages to retrieve")
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval mode (dense, bm25, hybrid)",
    )
    parser.add_argument("--hybrid-rrf-k", type=int, default=60, help="RRF k for hybrid fusion")
    
    args = parser.parse_args()

    # 0. Setup Config Overrides
    if args.lm_endpoint:
        global_config.set("vllm.endpoint", args.lm_endpoint)
    if args.lm_model:
        global_config.set("vllm.model", args.lm_model)

    # 1. Setup Workspace
    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    
    run_name = work_dir.name
    dataset_name = "mirage"
    
    setup_logging(str(work_dir / "run.log"))
    logger.info(f"Starting Vanilla RAG run in {work_dir}")
    cfg_snapshot = global_config.load_config()
    lm_endpoint = args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint")
    lm_model = args.lm_model or cfg_snapshot.get("vllm", {}).get("model")
    lm_temperature = (cfg_snapshot.get("vllm", {}) or {}).get("temperature", 0.0)
    emb_cfg = cfg_snapshot.get("retriever", {}).get("embedding", {})
    dataset_path = Path(args.dataset_path)
    doc_pool_path = dataset_path.parent / "doc_pool.json"

    cache_key: Optional[str] = None
    cache_path: Optional[str] = None
    cache_hit: Optional[bool] = None
    cache_meta: Dict[str, Any] = {}
    cache_enabled = args.index_path is None and args.chunk_store_path is None
    if cache_enabled:
        dataset_id = _infer_dataset_id(dataset_path)
        doc_pool_hash = _sha256_file(doc_pool_path) if doc_pool_path.exists() else "missing"
        cache_key, cache_meta = _build_cache_key(dataset_id, doc_pool_hash, emb_cfg, cfg_snapshot)
        cache_dir = Path(args.result_root) / "cache" / cache_key
        cache_index_path = cache_dir / "faiss.index"
        cache_chunk_store_path = cache_dir / "chunk_store.pkl"
        cache_embeddings_path = cache_dir / "embeddings.npy"
        cache_meta_path = cache_dir / "metadata.json"
        cache_build_log = cache_dir / "build.log"
        cache_required = [
            cache_index_path,
            cache_chunk_store_path,
            cache_embeddings_path,
            cache_meta_path,
            cache_build_log,
            cache_index_path.with_name(cache_index_path.name + ".meta.pkl"),
        ]
        cache_hit = all(path.exists() for path in cache_required)
        cache_path = str(cache_dir)
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=lm_model or "unknown",
            endpoint=lm_endpoint or "unknown",
            temperature=lm_temperature,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            decode={
                "temperature": lm_temperature,
                "top_p": None,
                "repetition_penalty": None,
                "max_tokens": args.max_new_tokens,
            },
            embedding={
                "model": emb_cfg.get("model"),
                "device": emb_cfg.get("device"),
                "batch_size": None,
                "max_length": emb_cfg.get("max_len_note"),
                "normalize": emb_cfg.get("normalize"),
                "dtype": emb_cfg.get("dtype"),
            },
            budgets={
                "context_budget_tokens": args.context_budget or None,
                "topk": args.topk,
            },
            extra={
                "retriever": args.retriever,
                "cache_key": cache_key,
                "cache_path": cache_path,
                "cache_hit": cache_hit,
            },
        ),
    )

    # 2. Determine Index Paths
    if cache_enabled:
        index_path = cache_index_path
        chunk_store_path = cache_chunk_store_path
    else:
        index_path = Path(args.index_path) if args.index_path else artifacts_dir / "vanilla_rag_index.faiss"
        chunk_store_path = Path(args.chunk_store_path) if args.chunk_store_path else artifacts_dir / "vanilla_rag_chunk_store.pkl"

    # 3. Check/Build Index
    if cache_enabled:
        if cache_hit:
            logger.info("Embedding cache hit: {} -> {}", cache_key, cache_path)
        else:
            logger.info("Embedding cache miss: {} -> {}", cache_key, cache_path)
            if not doc_pool_path.exists():
                logger.error(f"Cannot build index: doc_pool.json not found at {doc_pool_path}")
                return
            logger.info(f"Building index from {doc_pool_path}...")
            try:
                docs = _load_doc_pool(doc_pool_path)
            except Exception as e:
                logger.error(f"Failed to load doc pool: {e}")
                return

            if not docs:
                logger.error("No documents found to index.")
                return

            build_started = time.time()
            indexer = VanillaRAGIndexer(cfg_snapshot)
            build_info = indexer.build(
                docs,
                str(cache_index_path),
                str(cache_chunk_store_path),
                output_embeddings_path=str(cache_embeddings_path),
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            meta_payload = {
                **cache_meta,
                "cache_key": cache_key,
                "cache_path": cache_path,
                "dataset_path": str(dataset_path),
                "doc_pool_path": str(doc_pool_path),
                "doc_pool_hash": cache_meta.get("doc_pool_hash"),
                "build": build_info,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            cache_meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            duration_s = time.time() - build_started
            cache_build_log.write_text(
                f"cache_key={cache_key}\n"
                f"dataset_path={dataset_path}\n"
                f"doc_pool_path={doc_pool_path}\n"
                f"chunks={build_info.get('chunk_count')}\n"
                f"vectors={build_info.get('vector_count')}\n"
                f"dim={build_info.get('vector_dim')}\n"
                f"duration_s={duration_s:.2f}\n",
                encoding="utf-8",
            )
            logger.info("Index built successfully and cached.")
    else:
        if not index_path.exists() or not chunk_store_path.exists():
            logger.info(f"Index not found at {index_path}. Attempting to build...")

            if not doc_pool_path.exists():
                logger.error(f"Cannot build index: doc_pool.json not found at {doc_pool_path}")
                return

            logger.info(f"Building index from {doc_pool_path}...")
            try:
                docs = _load_doc_pool(doc_pool_path)
            except Exception as e:
                logger.error(f"Failed to load doc pool: {e}")
                return

            if not docs:
                logger.error("No documents found to index.")
                return

            indexer = VanillaRAGIndexer(cfg_snapshot)
            indexer.build(docs, str(index_path), str(chunk_store_path))
            logger.info("Index built successfully.")
    retriever = None
    if args.retriever in ("dense", "hybrid"):
        retriever = get_retriever(
            index_path=str(index_path),
            chunk_store_path=str(chunk_store_path),
            context_budget=args.context_budget,
        )
    llm = get_default_llm_client(llm_profile="generate")

    bm25_index = None
    bm25_ids: List[str] = []
    bm25_texts: List[str] = []
    bm25_id_to_idx: Dict[str, int] = {}
    if args.retriever in ("bm25", "hybrid"):
        try:
            with open(chunk_store_path, "rb") as f:
                chunk_store = pickle.load(f)
            meta_path = index_path.with_name(index_path.name + ".meta.pkl")
            with open(meta_path, "rb") as f:
                chunk_ids = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunk store/meta for BM25: {e}")
            return
        for cid in chunk_ids:
            text = chunk_store.get(cid)
            if not text:
                continue
            bm25_ids.append(str(cid))
            bm25_texts.append(str(text))
        bm25_id_to_idx = {cid: idx for idx, cid in enumerate(bm25_ids)}
        bm25_index = BM25Index(bm25_texts)

    # 4. Load Dataset
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    pred_raw_records: List[Dict[str, Any]] = []

    def _bm25_hit(idx: int, score: float) -> Dict[str, Any]:
        cid = bm25_ids[idx]
        text = bm25_texts[idx]
        doc_id = cid.split("::", 1)[0] if "::" in cid else None
        return {
            "text": text,
            "score": float(score),
            "doc_id": doc_id,
            "sent_ids": None,
            "passage_id": cid,
        }
    
    # 5. Run Inference
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            hits: List[Dict[str, Any]] = []
            if args.retriever == "dense":
                if retriever is None:
                    raise RuntimeError("Dense retriever unavailable.")
                hits = retriever.retrieve(question, top_k=args.topk)
            elif args.retriever == "bm25":
                if bm25_index is None:
                    raise RuntimeError("BM25 index unavailable.")
                scores = bm25_index.get_scores(question)
                indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.topk]
                hits = [_bm25_hit(idx, scores[idx]) for idx in indices]
            else:
                if retriever is None or bm25_index is None:
                    raise RuntimeError("Hybrid retriever unavailable.")
                dense_hits = retriever.retrieve(question, top_k=args.topk)
                bm25_scores = bm25_index.get_scores(question)
                bm25_rank = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
                dense_rank = [
                    bm25_id_to_idx[hit["passage_id"]]
                    for hit in dense_hits
                    if hit.get("passage_id") in bm25_id_to_idx
                ]
                fused = rrf_fuse([dense_rank, bm25_rank], k=int(getattr(args, "hybrid_rrf_k", 60)))
                indices = sorted(fused, key=fused.get, reverse=True)[: args.topk]
                hits = [_bm25_hit(idx, fused.get(idx, 0.0)) for idx in indices]

            annotated_hits = []
            for idx, hit in enumerate(hits):
                annotated_hits.append({**hit, "text": f"[{idx+1}] {hit.get('text', '')}"})
            context_str, contexts_used, context_tokens = pack_contexts(
                annotated_hits, int(args.context_budget or 0)
            )
            prompt = f"""Answer the question based on the context.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""
            ans = llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 1024,
            )
            final_ans = ans
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans
            })
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=[
                        {**hit, "rank": idx + 1} for idx, hit in enumerate(hits)
                    ],
                    topk=len(hits),
                    final_context=contexts_used,
                    final_context_tokens=context_tokens,
                    context_budget_tokens=int(args.context_budget or 0) or None,
                    log_dir=artifacts_dir,
                )
            except Exception as log_exc:
                logger.error(f"retrieval logging failed for {qid}: {log_exc}")
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": ans,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": int(args.context_budget or 0) or None,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 6. Save Results
    # Save detailed JSONL
    out_jsonl = preds_dir / "results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    # Save QA TSV (compatible with eval scripts)
    out_tsv = preds_dir / "qa.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        for res in results:
            # Format: query_text\tmodel_answer
            q_text = res["question"].replace("\t", " ").strip()
            a_text = res["answer"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q_text}\t{a_text}\n")
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)

    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
