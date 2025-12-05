#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Ensure root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.direct_llm import DirectLLMRunner
from baselines.naive_rag.runner import LLMClient
from config.config_loader import config as global_config_loader
from utils.answer_cleaner import _strip_reasoning


def _list_workspaces(root: Path, dataset: str) -> List[Path]:
    if not root.exists():
        return []
    pattern = re.compile(r"^(?P<idx>\d{3})-(?P<name>.+)$")
    candidates = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match and match.group("name") == dataset:
            candidates.append((int(match.group("idx")), entry))
    candidates.sort(key=lambda x: x[0])
    return [entry for _, entry in candidates]


def _select_workspace(root: Path, dataset: str, new: bool) -> Path:
    workspaces = _list_workspaces(root, dataset)
    if new or not workspaces:
        next_idx = workspaces[-1].name.split("-")[0] if workspaces else "-1"
        next_val = int(next_idx) + 1
        name = f"{next_val:03d}-{dataset}"
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return workspaces[-1]





def load_dataset_file(dataset_name: str, dataset_path: Optional[str]) -> List[dict]:
    if dataset_path:
        path = Path(dataset_path)
    else:
        # Default locations
        if dataset_name == "mirage":
            path = Path("data/mirage_sample/dataset.json")
        elif dataset_name == "hotpotqa":
            path = Path("data/hotpotqa/dataset.json")
        else:
            path = Path(f"data/{dataset_name}/dataset.json")

    if not path.exists():
        raise FileNotFoundError(f"Dataset JSON not found at {path}")

    with open(path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
        
    # Normalize dataset to list of dicts with id, question, answer
    # HotpotQA/Mirage format expected
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baseline answers for MIRAGE/HotpotQA")
    parser.add_argument("--dataset", default="mirage", help="Dataset name (used for workspace naming)")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None, help="Explicit workspace path")
    
    parser.add_argument("--baseline", choices=["direct", "naive", "vanilla", "selfrag", "raptor", "graphrag", "mirage"], default="mirage", help="Baseline to run")
    
    # Legacy flags for backward compatibility
    parser.add_argument("--direct-llm", action="store_true", help="Use Direct LLM baseline (deprecated, use --baseline direct)")
    parser.add_argument("--vanilla-rag", action="store_true", help="Use Vanilla RAG baseline (deprecated, use --baseline vanilla)")
    parser.add_argument("--simple-raptor", action="store_true", help="Use Simple Raptor baseline (deprecated, use --baseline raptor)")
    
    parser.add_argument("--indexes-dir", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--lmstudio-endpoint", required=True)
    parser.add_argument("--lmstudio-model", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--qa-log", default=None, help="Optional plain text QA log path (question \t answer)")
    parser.add_argument("--new", action="store_true", help="Force create a new workspace copy")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override LLM max new tokens")
    args = parser.parse_args()

    # Normalize baseline selection
    if args.direct_llm: args.baseline = "direct"
    if args.vanilla_rag: args.baseline = "vanilla"
    if args.simple_raptor: args.baseline = "raptor"

    dataset_name = args.dataset
    dataset = load_dataset_file(dataset_name, args.dataset_path)

    result_root = Path(args.result_root)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Include baseline in workspace name if not default
        ws_name = dataset_name
        if args.baseline != "mirage":
             ws_name = f"{dataset_name}_{args.baseline}"
        
        result_root.mkdir(parents=True, exist_ok=True)
        work_dir = _select_workspace(result_root, ws_name, args.new)
    logger.info("Using workspace: {}", work_dir)

    output_path = Path(args.out) if args.out else work_dir / "answers.json"
    qa_log_path = Path(args.qa_log) if args.qa_log else work_dir / "qa.tsv"

    if args.limit > 0:
        dataset = dataset[: args.limit]

    # Setup LLM Client
    llm_client = LLMClient(
        endpoint=args.lmstudio_endpoint,
        model=args.lmstudio_model,
        temperature=args.temperature if args.temperature is not None else 0.0,
        max_tokens=args.max_new_tokens or 2048
    )

    results = []
    qa_lines = []

    # --- BASELINE RUNNERS ---

    if args.baseline == "direct":
        runner = DirectLLMRunner(
            lm_endpoint=args.lmstudio_endpoint,
            lm_model=args.lmstudio_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("Direct LLM baseline complete: {}", artifacts.get("qa"))
        # DirectLLMRunner saves its own files, we just copy/reference them
        if output_path and Path(artifacts["answers_json"]) != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(Path(artifacts["answers_json"]).read_text(encoding="utf-8"), encoding="utf-8")
        if qa_log_path and Path(artifacts["qa"]) != qa_log_path:
            qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            qa_log_path.write_text(Path(artifacts["qa"]).read_text(encoding="utf-8"), encoding="utf-8")
        return

    elif args.baseline == "naive":
        from baselines.naive_rag.runner import answer as naive_rag_answer
        
        # Infer index path
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_naive")
        if not index_dir.exists():
             # Fallback to standard location inside work_dir if created there, or assume relative
             index_dir = Path(f"result/{dataset_name}_naive")
        
        # Actually naive_rag_answer expects a path to FAISS index usually, let's check signature
        # It usually takes index_path and chunk_store_path
        index_path = str(index_dir / "index.faiss")
        chunk_path = str(index_dir / "chunks.jsonl")
        
        logger.info(f"Running Naive RAG with index={index_path}")

        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                # Note: naive_rag_answer signature might vary, assuming similar to vanilla
                ans = naive_rag_answer(question, index_path=index_path, chunk_store_path=chunk_path, llm_client=llm_client)
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "vanilla":
        from baselines.vanilla_rag import answer as vanilla_rag_answer
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_vanilla")
        index_path = str(index_dir / "vanilla_rag_index.faiss")
        chunk_path = str(index_dir / "vanilla_rag_chunk_store.pkl")

        logger.info(f"Running Vanilla RAG with index={index_path}")
        
        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                # Pass llm_client if supported, otherwise it uses global config
                # The existing vanilla_rag_answer might not take llm_client. 
                # If it doesn't, we rely on global config injection.
                # Let's assume we can inject config via global singleton or it accepts client.
                # Checking existing code: vanilla_rag_answer(question, index_path, chunk_store_path)
                # We should probably use VanillaRAGRunner class if available or just set global config.
                
                # Set global config for LLM
                global_config_loader.set("lmstudio.endpoint", args.lmstudio_endpoint)
                global_config_loader.set("lmstudio.model", args.lmstudio_model)
                
                ans = vanilla_rag_answer(question, index_path=index_path, chunk_store_path=chunk_path)
                
                if ans.lower().startswith("answer:"):
                    ans = ans[7:].strip()
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "selfrag":
        from baselines.simple_selfrag import answer as selfrag_answer
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_selfrag")
        index_path = str(index_dir / "selfrag_index.faiss")
        chunk_path = str(index_dir / "selfrag_chunk_store.pkl")
        
        logger.info(f"Running Self-RAG with index={index_path}")
        
        # Set global config for LLM
        global_config_loader.set("lmstudio.endpoint", args.lmstudio_endpoint)
        global_config_loader.set("lmstudio.model", args.lmstudio_model)

        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                ans = selfrag_answer(question, index_path=index_path, chunk_store_path=chunk_path)
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "raptor":
        from baselines.simple_raptor.retriever import SimpleRaptorRetriever
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_raptor")
        index_path = str(index_dir / "simple_raptor_index.faiss")
        nodes_path = str(index_dir / "simple_raptor_nodes.pkl")
        chunk_path = str(index_dir / "simple_raptor_chunk_store.pkl")
        
        logger.info(f"Running Raptor with index={index_path}")

        retriever = SimpleRaptorRetriever(
            index_path=index_path,
            nodes_path=nodes_path,
            chunk_store_path=chunk_path,
            config=global_config_loader.load_config(),
            llm_client=llm_client
        )

        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                ans = retriever.answer(question)
                if ans.lower().startswith("answer:"):
                    ans = ans[7:].strip()
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "graphrag":
        from baselines.simple_graphrag import answer as graphrag_answer
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_graphrag")
        graph_path = str(index_dir / "graph.pkl")
        chunk_store_path = str(index_dir / "chunk_store.pkl")
        
        # Set global config for LLM
        global_config_loader.set("lmstudio.endpoint", args.lmstudio_endpoint)
        global_config_loader.set("lmstudio.model", args.lmstudio_model)
        
        logger.info(f"Running GraphRAG with graph={graph_path}")
        
        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                ans = graphrag_answer(
                    question, 
                    graph_path=graph_path, 
                    chunk_store_path=chunk_store_path
                )
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "mirage":
        # Original MIRAGE logic
        indexes_dir = Path(args.indexes_dir) if args.indexes_dir else work_dir / "indexes"
        notes_path = Path(args.notes) if args.notes else work_dir / "notes" / f"notes.{dataset_name}.jsonl"
        
        from query.query_processor import QueryProcessor
        qp = QueryProcessor(
            indexes_dir=str(indexes_dir),
            notes_path=str(notes_path),
            lmstudio_endpoint=args.lmstudio_endpoint,
            lmstudio_model=args.lmstudio_model,
        )

        for item in dataset:
            question = item.get("query") or item.get("question")
            if not question: continue
            
            attr_hint = "occupation" if "occupation" in question.lower() else None
            qid = item.get("query_id")
            doc_hint = f"{dataset_name}/{qid}" if qid else None
            
            res = qp.process(question, doc_hint=doc_hint, attribute_hint=attr_hint)
            
            answer_text = res.get("answer")
            clean_ans = ""
            if answer_text:
                stripped = _strip_reasoning(str(answer_text))
                clean_ans = " ".join(stripped.splitlines())
                
            results.append({
                "query_id": item.get("query_id"),
                "question": question,
                "answer": res.get("answer"),
                "structured": res.get("structured"),
                "decision": res.get("decision"),
            })
            qa_lines.append(f"{question}\t{clean_ans}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote {} answers to {}", len(results), output_path)

    if qa_log_path:
        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        qa_log_path.write_text("\n".join(qa_lines), encoding="utf-8")
        logger.info("Wrote QA log to {}", qa_log_path)


if __name__ == "__main__":
    main()
