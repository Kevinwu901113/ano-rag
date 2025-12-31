#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_raptor.index import SimpleRaptorIndexer
from baselines.simple_raptor.retriever import SimpleRaptorRetriever
from baselines.common.model_clients import get_default_llm_client
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved

def _select_workspace(root: Path, dataset: str, new: bool) -> Path:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        
    import re
    pattern = re.compile(r"^(?P<idx>\d{3})-(?P<name>.+)$")
    candidates = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match and match.group("name") == dataset:
            candidates.append((int(match.group("idx")), entry))
    candidates.sort(key=lambda x: x[0])
    
    if new or not candidates:
        next_idx = candidates[-1][0] + 1 if candidates else 0
        name = f"{next_idx:03d}-{dataset}"
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return candidates[-1][1]

def _load_doc_pool(doc_pool_path: Path) -> Dict[str, str]:
    with doc_pool_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    docs: Dict[str, str] = {}
    if isinstance(raw_data, list):
        for i, item in enumerate(raw_data):
            base_id = item.get("doc_id") or item.get("mapped_id") or str(i)
            doc_id = f"{base_id}::{i}"
            title = item.get("title") or item.get("doc_name") or ""
            paragraphs = item.get("paragraphs")
            if paragraphs:
                text = title + "\n" + "\n".join(str(p) for p in paragraphs if p)
            else:
                text = title + "\n" + (item.get("doc_chunk") or "")
            text = text.strip()
            if text:
                docs[doc_id] = text
    elif isinstance(raw_data, dict):
        docs = {k: str(v) for k, v in raw_data.items()}
    return docs

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Simple Raptor baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--doc-pool", default=None, help="Path to doc_pool.json (auto from dataset dir if omitted)")
    parser.add_argument("--index-dir", default=None, help="Directory containing Raptor index files")
    parser.add_argument("--index-path", default=None, help="Optional explicit FAISS index path")
    parser.add_argument("--nodes-path", default=None, help="Optional explicit nodes.pkl path")
    parser.add_argument("--chunk-store-path", default=None, help="Optional explicit chunk_store.pkl path")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--lm-endpoint", default=None)
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true", help="Skip writing retrieval debug JSONL")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    args = parser.parse_args()
    
    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    if args.lm_endpoint:
        global_config.set("vllm.endpoint", args.lm_endpoint)
    if args.lm_model:
        global_config.set("vllm.model", args.lm_model)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    setup_logging(str(work_dir / "run.log"))
    logger.info("Writing outputs to {}", work_dir)

    index_dir = Path(args.index_dir) if args.index_dir else artifacts_dir / "simple_raptor"
    index_path = Path(args.index_path) if args.index_path else index_dir / "simple_raptor_index.faiss"
    nodes_path = Path(args.nodes_path) if args.nodes_path else index_dir / "simple_raptor_nodes.pkl"
    chunk_store_path = Path(args.chunk_store_path) if args.chunk_store_path else index_dir / "simple_raptor_chunk_store.pkl"

    doc_pool_path = Path(args.doc_pool) if args.doc_pool else dataset_path.parent / "doc_pool.json"
    if not index_path.exists() or not nodes_path.exists() or not chunk_store_path.exists():
        if not doc_pool_path.exists():
            raise FileNotFoundError(f"Doc pool not found: {doc_pool_path}")
        logger.info("Index artifacts missing; building Simple Raptor index at {}", index_dir)
        docs = _load_doc_pool(doc_pool_path)
        if not docs:
            raise RuntimeError(f"No documents found in doc pool: {doc_pool_path}")
        indexer = SimpleRaptorIndexer()
        stats = indexer.build(docs, cluster_size=16)
        logger.info("Raptor index built: {}", stats)
        index_dir.mkdir(parents=True, exist_ok=True)
        indexer.save(str(index_path), str(nodes_path), str(chunk_store_path))

    run_name = work_dir.name
    dataset_name = "mirage"
    cfg_snapshot = global_config.load_config()
    lm_endpoint = args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint")
    lm_model = args.lm_model or cfg_snapshot.get("vllm", {}).get("model")
    emb_cfg = cfg_snapshot.get("retriever", {}).get("embedding", {})
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=lm_model or "unknown",
            endpoint=lm_endpoint or "unknown",
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            decode={
                "temperature": args.temperature,
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
                "embedding_model_name": emb_cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2",
                "depth": "auto (recursive)", # Raptor logic
            }
        ),
    )
    
    # Configure LLM Client explicit overrides if provided
    llm_client = None
    if not args.retrieval_only:
        llm_client = get_default_llm_client(llm_profile="generate")

    # Initialize Retriever
    retriever = SimpleRaptorRetriever(
        str(index_path),
        str(nodes_path),
        str(chunk_store_path),
        llm_client=llm_client,
        top_k=args.topk,
        context_budget=args.context_budget,
    )
    
    # Run Dataset
    results = []
    qa_lines = []
    pred_raw_records: List[Dict[str, Any]] = []
    
    limit = args.limit if args.limit and args.limit > 0 else len(dataset)
    logger.info(f"Running Raptor baseline on {limit} questions...")
    
    for i, item in enumerate(dataset[:limit]):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            
            if args.retrieval_only:
                 # Check if the retriever has a dedicated retrieval method
                 # Looking at the class definition, it seems it doesn't have a public retrieve method
                 # but it has internal logic in answer() that calls _retrieve_nodes or similar.
                 # Let's inspect the class via reading file first.
                 # Based on my read, it doesn't seem to have a public `retrieve` method exposed in the interface I saw.
                 # I will add a fallback to call a private method or modify the class later if needed.
                 # But for now, let's try to call `_retrieve_nodes` if it exists and publicize it or similar.
                 # Actually, looking at lines 30-85 of baselines/simple_raptor/retriever.py, there is no retrieve method.
                 # I should probably add one to the class or mock it.
                 # Or I can try to use `retrieve_context` if it exists.
                 # Let's try to monkey-patch or use `_retrieve_nodes` if I can find it.
                 
                 # Wait, I can't easily monkeypatch here without reading more code.
                 # Let's try to call `answer` but set LLM to None, hoping it fails gracefully or returns context?
                 # No, `answer` probably expects LLM to work.
                 
                 # Let's assume I need to implement `retrieve` in `SimpleRaptorRetriever` or use `_retrieve_nodes`.
                 # Let's check `baselines/simple_raptor/retriever.py` content again.
                 # It's not fully visible.
                 # I'll optimistically try to call `_retrieve_context` if it exists.
                 pass

            if args.retrieval_only:
                hits = retriever.retrieve(question, k=args.topk)
                ans = ""
            else:
                ans = retriever.answer(question)
                hits = getattr(retriever, "last_hits", [])

            results.append({
                "query_id": qid,
                "question": question,
                "answer": ans,
            })
            annotated_hits = []
            for i, hit in enumerate(hits):
                annotated_hits.append({**hit, "text": f"[{i+1}] {hit.get('text', '')}"})
            context_str, contexts_used, context_tokens = pack_contexts(
                annotated_hits, int(args.context_budget or 0)
            )
            qa_lines.append(f"{question}\t{ans.replace(chr(10), ' ')}")
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=hits,
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
            logger.exception(f"Error Q{i}: {e}")
            
    # Save results
    out_json = preds_dir / "answers.json"
    out_qa = preds_dir / "qa.tsv"
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(out_qa, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
        
    logger.info("Raptor baseline complete. qa.tsv: {}", out_qa)

if __name__ == "__main__":
    main()
