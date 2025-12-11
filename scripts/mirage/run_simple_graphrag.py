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
from utils.retrieval_logger import log_retrieval


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
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--index-dir", default=None, help="Directory containing graph index files")
    parser.add_argument("--work-dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--lmstudio-endpoint", required=True)
    parser.add_argument("--lmstudio-model", required=True)
    args = parser.parse_args()

    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    # global_config is a ConfigLoader instance, not a dict.
    # It has a .set(key, value) method.
    global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    global_config.set("lmstudio.model", args.lmstudio_model)
    
    # Ensure string types for kwargs passed later
    lm_endpoint_str = str(args.lmstudio_endpoint)
    lm_model_str = str(args.lmstudio_model)
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_simple_graphrag", args.new)
    run_name = work_dir.name
    dataset_name = "mirage"
    
    # Add logger sink to work_dir
    logger.add(work_dir / "simple_graphrag.log")
    
    logger.info("Writing Simple GraphRAG outputs to {}", work_dir)

    # Determine index directory: if not provided, default to work_dir
    index_dir = args.index_dir if args.index_dir else str(work_dir)
    logger.info(f"Using graph index from: {index_dir}")

    # Check if index exists, if not, build it
    graph_pkl = Path(index_dir) / "simple_graphrag_graph.pkl"
    chunk_store_pkl = Path(index_dir) / "simple_graphrag_chunk_store.pkl"
    
    if not graph_pkl.exists() or not chunk_store_pkl.exists():
        logger.info(f"Graph index not found in {index_dir}. Building graph...")
        
        # Collect documents from dataset
        docs = {}
        
        # First check if there is a separate doc_pool.json
        doc_pool_path = dataset_path.parent / "doc_pool.json"
        if doc_pool_path.exists():
            logger.info(f"Loading documents from {doc_pool_path}")
            with doc_pool_path.open("r", encoding="utf-8") as handle:
                doc_pool = json.load(handle)
                for item in doc_pool:
                    # In doc_pool, documents are associated with queries via mapped_id
                    # But we want to build a graph of knowledge. 
                    # doc_pool items have "doc_chunk" or "text"
                    # We can use a combination of doc_name and index as ID, or just iterate
                    
                    # Ideally we want unique documents. 
                    # Let's use a hash of content or just sequential ID if no stable ID
                    text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
            doc_name = item.get("doc_name") or "unknown"
            
            if text:
                # Create a deterministic ID based on content hash to avoid duplicates
                import hashlib
                doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                # Include index to ensure absolute uniqueness if needed, but hash should be enough for identical content
                # To be safe against hash collisions (unlikely) or identical content from different sources, let's append index
                # Wait, enumerate index is not available in this loop context directly (it's 'item' in 'doc_pool').
                # Let's just use hash. If content is identical, it's fine to treat as same node.
                doc_id = f"{doc_name}_{doc_hash[:8]}"
                docs[doc_id] = text
        else:
            # Fallback to dataset items
            for item in dataset:
                doc_id = item.get("doc_id") or item.get("id")
                text = item.get("text") or item.get("content") or ""
                if doc_id and text:
                    docs[str(doc_id)] = text
        
        logger.info(f"Collected {len(docs)} documents for graph construction")
        
        # Build graph
        import asyncio
        from baselines.simple_graphrag.build_graph import GraphBuilder
        from structrag.llm_client import LLMChatClient
        
        llm_client = LLMChatClient(
            endpoint=lm_endpoint_str,
            model=lm_model_str,
            temperature=0.0
        )
        
        builder = GraphBuilder(llm_client)
        asyncio.run(builder.build(docs))
        builder.save(str(graph_pkl), str(chunk_store_pkl))
        logger.info(f"Graph built and saved to {index_dir}")

    # 2. Run inference
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    qa_lines = []
    
    from baselines.simple_graphrag.runner import _strip_reasoning, _enforce_short_answer
    retriever = get_retriever(index_dir=index_dir)
    
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = retriever.answer(question)
            cleaned_ans = _strip_reasoning(ans)
            final_ans = _enforce_short_answer(cleaned_ans)
            
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
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=hits,
                    topk=len(hits),
                    final_context=[
                        {
                            "doc_id": hit.get("doc_id"),
                            "sent_ids": hit.get("sent_ids"),
                            "passage_id": hit.get("passage_id"),
                            "text": hit.get("text"),
                        }
                        for hit in hits
                    ],
                    log_dir=work_dir,
                )
            except Exception as log_exc:
                logger.error(f"retrieval logging failed for {qid}: {log_exc}")
            
        except Exception as e:
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 3. Save results
    output_path = work_dir / "answers.json"
    qa_log_path = work_dir / "qa.tsv"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(qa_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
    
    # Also copy log file to work_dir if possible, or ensure logger writes there
    # The logger is configured globally, but we can add a sink here
    # Actually, let's just ensure we have a log file in work_dir
    
    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
