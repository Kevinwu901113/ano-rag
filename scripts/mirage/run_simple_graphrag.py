#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_graphrag import get_retriever
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved


def _select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        next_idx = len(existing)
        target = root / f"{prefix}_{next_idx:03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Simple GraphRAG baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--index-dir", default=None, help="Directory containing graph index files")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--lm-endpoint", default=None)
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    args = parser.parse_args()

    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    # global_config is a ConfigLoader instance, not a dict.
    # It has a .set(key, value) method.
    if args.lm_endpoint:
        global_config.set("vllm.endpoint", args.lm_endpoint)
    if args.lm_model:
        global_config.set("vllm.model", args.lm_model)
    
    cfg_snapshot = global_config.load_config()
    lm_endpoint_str = str(args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint") or "")
    lm_model_str = str(args.lm_model or cfg_snapshot.get("vllm", {}).get("model") or "")
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    dataset_name = "mirage"
    
    setup_logging(str(work_dir / "run.log"))
    logger.info("Writing Simple GraphRAG outputs to {}", work_dir)
    emb_cfg = cfg_snapshot.get("retriever", {}).get("embedding", {})
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=lm_model_str or "unknown",
            endpoint=lm_endpoint_str or "unknown",
            temperature=None,
            max_tokens=None,
            context_budget=args.context_budget or None,
            embedding={
                "model": emb_cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2",
                "device": emb_cfg.get("device") or "cpu",
            },
            extra={
                "embedding_model_name": emb_cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2",
                "expand_hop": 1,
                "n_expand": 5, # Implicit in logic if not configurable
            }
        ),
    )

    # Determine index directory: if not provided, default to work_dir
    index_dir = args.index_dir if args.index_dir else str(artifacts_dir / "simple_graphrag")
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using graph index from: {index_dir}")

    # Paths for Graph and Dense Index
    graph_pkl = Path(index_dir) / "simple_graphrag_graph.pkl"
    chunk_store_pkl = Path(index_dir) / "simple_graphrag_chunk_store.pkl"
    
    dense_index_dir = artifacts_dir / "dense_index"
    dense_index_dir.mkdir(parents=True, exist_ok=True)
    dense_index_path = dense_index_dir / "faiss.index"
    dense_chunk_store_path = dense_index_dir / "chunk_store.pkl"
    
    # Check what is missing
    missing_graph = not graph_pkl.exists() or not chunk_store_pkl.exists()
    missing_dense = not dense_index_path.exists() or not dense_chunk_store_path.exists()
    
    docs = {}
    if missing_graph or missing_dense:
        logger.info(f"Artifacts missing (Graph: {missing_graph}, Dense: {missing_dense}). Loading docs...")
        
        # Collect documents from dataset
        # First check if there is a separate doc_pool.json
        doc_pool_path = dataset_path.parent / "doc_pool.json"
        if doc_pool_path.exists():
            logger.info(f"Loading documents from {doc_pool_path}")
            with doc_pool_path.open("r", encoding="utf-8") as handle:
                doc_pool = json.load(handle)
                for idx, item in enumerate(doc_pool):
                    text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
                    doc_name = item.get("doc_name") or "unknown"
            
                    if text:
                        # Use mapped_id if available to satisfy Canonical ID requirement
                        base_id = item.get("mapped_id") or item.get("doc_id")
                        if base_id:
                            doc_id = f"{base_id}::{idx}"
                        else:
                            # Fallback to hash if no ID
                            import hashlib
                            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                            doc_id = f"{doc_name}_{doc_hash[:8]}"
                            
                        docs[doc_id] = text
        else:
            # Fallback to dataset items
            for item in dataset:
                doc_id = item.get("doc_id") or item.get("id")
                text = item.get("text") or item.get("content") or ""
                if doc_id and text:
                    docs[str(doc_id)] = text
        
        logger.info(f"Collected {len(docs)} documents")

    # Build Graph if missing
    if missing_graph:
        logger.info(f"Graph index not found in {index_dir}. Building graph...")
        import asyncio
        from baselines.simple_graphrag.build_graph import GraphBuilder
        from baselines.common.model_clients import get_default_llm_client
        
        llm_client = get_default_llm_client(global_config.load_config())
        
        builder = GraphBuilder(llm_client)
        asyncio.run(builder.build(docs))
        builder.save(str(graph_pkl), str(chunk_store_pkl))
        logger.info(f"Graph built and saved to {index_dir}")

    # Build Dense Index if missing
    if missing_dense:
        logger.info("Dense index not found. Building for LightRAG seeds...")
        from baselines.vanilla_rag.index import VanillaRAGIndexer
        if docs:
            indexer = VanillaRAGIndexer(cfg_snapshot)
            indexer.build(docs, str(dense_index_path), str(dense_chunk_store_path))
        else:
            logger.warning("No docs found to build dense index!")

    # 2. Run inference
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    qa_lines = []
    pred_raw_records: List[Dict[str, Any]] = []
    
    retriever = get_retriever(index_dir=index_dir, context_budget=args.context_budget, dense_index_path=str(dense_index_path))
    
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = retriever.answer(question)
            final_ans = ans.strip()
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans
            })
            
            clean_ans = " ".join(final_ans.split())
            qa_lines.append(f"{question}\t{clean_ans}")
            try:
                hits = getattr(retriever, "last_hits", [])
                annotated_hits = []
                for idx, hit in enumerate(hits):
                    annotated_hits.append({**hit, "text": f"[{idx+1}] {hit.get('text', '')}"})
                context_str, contexts_used, context_tokens = pack_contexts(
                    annotated_hits, int(args.context_budget or 0)
                )
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
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 3. Save results
    output_path = preds_dir / "answers.json"
    qa_log_path = preds_dir / "qa.tsv"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(qa_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
    
    # Also copy log file to work_dir if possible, or ensure logger writes there
    # The logger is configured globally, but we can add a sink here
    # Actually, let's just ensure we have a log file in work_dir
    
    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
