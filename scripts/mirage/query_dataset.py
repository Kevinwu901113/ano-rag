#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from loguru import logger
from baselines.direct_llm import DirectLLMRunner
from utils.run_layout import ensure_workdir_layout, resolve_workdir


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


def _strip_reasoning(answer: str) -> str:
    text = answer
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start + 7)
        if end == -1:
            text = text[:start] + text[start + 7 :]
            break
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM answers for MIRAGE dataset")
    parser.add_argument("--dataset", default="mirage", help="Dataset name (used for workspace naming)")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset.json")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Explicit workspace path")
    parser.add_argument("--indexes-dir", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--lm-endpoint", default=None)
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--qa-log", default=None, help="Optional plain text QA log path (question \t answer)")
    parser.add_argument("--new", action="store_true", help="Force create a new workspace copy")
    parser.add_argument("--direct-llm", action="store_true", help="Skip retrieval and answer directly with LLM")
    parser.add_argument("--vanilla-rag", action="store_true", help="Use Vanilla RAG baseline")
    parser.add_argument("--simple-raptor", action="store_true", help="Use Simple Raptor baseline")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override LLM max new tokens")
    args = parser.parse_args()

    from config.config_loader import config as global_config_loader
    cfg_snapshot = global_config_loader.load_config()
    lm_endpoint = args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint")
    lm_model = args.lm_model or cfg_snapshot.get("vllm", {}).get("model")

    dataset_name = args.dataset
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(f"data/{dataset_name}_sample/dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found at {dataset_path}")

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset=dataset_name)
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    logger.info("Using workspace: {}", work_dir)

    override_cfg = work_dir / "config.override.yaml"
    if override_cfg.exists():
        os.environ["ANO_RAG_CONFIG"] = str(override_cfg)

    output_path = Path(args.out) if args.out else preds_dir / "answers.json"
    qa_log_path = Path(args.qa_log) if args.qa_log else preds_dir / "qa.tsv"

    with open(dataset_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.limit > 0:
        dataset = dataset[: args.limit]

    if args.direct_llm:
        runner = DirectLLMRunner(
            lm_endpoint=lm_endpoint,
            lm_model=lm_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("Direct LLM baseline complete: {}", artifacts.get("qa"))
        if output_path and Path(artifacts["answers_json"]) != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(Path(artifacts["answers_json"]).read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Copied answers.json to {}", output_path)
        if qa_log_path and Path(artifacts["qa"]) != qa_log_path:
            qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            qa_log_path.write_text(Path(artifacts["qa"]).read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Copied QA log to {}", qa_log_path)
        return

    if args.vanilla_rag:
        from baselines.vanilla_rag import answer as vanilla_rag_answer
        
        # Assume vanilla rag index/chunks are in standard location or passed via args
        # We can reuse indexes-dir to point to directory containing vanilla_rag_index.faiss
        index_path = "indexes/vanilla_rag_index.faiss"
        chunk_path = "indexes/vanilla_rag_chunk_store.pkl"
        
        if args.indexes_dir:
             idx_root = Path(args.indexes_dir)
             if (idx_root / "vanilla_rag_index.faiss").exists():
                 index_path = str(idx_root / "vanilla_rag_index.faiss")
                 chunk_path = str(idx_root / "vanilla_rag_chunk_store.pkl")
             elif idx_root.is_file(): # user pointed directly to index
                 index_path = str(idx_root)
                 # try to guess chunk store
                 chunk_path = str(idx_root.parent / "vanilla_rag_chunk_store.pkl")

        logger.info(f"Running Vanilla RAG with index={index_path}")
        
        results = []
        qa_lines = []
        
        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            
            try:
                ans = vanilla_rag_answer(question, index_path=index_path, chunk_store_path=chunk_path)
                # Simple cleaning
                if ans.lower().startswith("answer:"):
                    ans = ans[7:].strip()
                clean_ans = _strip_reasoning(ans)
                clean_ans_line = " ".join(clean_ans.split())
                
                results.append({
                    "query_id": qid,
                    "question": question,
                    "answer": clean_ans,
                    "raw_answer": ans
                })
                qa_lines.append(f"{question}\t{clean_ans_line}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")
                
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        
        if qa_log_path:
            qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            qa_log_path.write_text("\n".join(qa_lines), encoding="utf-8")
            
        logger.info("Vanilla RAG baseline complete.")
        return

    if args.simple_raptor:
        from baselines.simple_raptor import answer as raptor_answer
        from baselines.simple_raptor import get_retriever

        # For Raptor, we might need to init the retriever explicitly if we want to pass LLM config
        # But the simple_raptor.answer() wrapper uses a global singleton that loads global config.
        # To support CLI overrides (endpoint/model), we should instantiate it manually here.
        from baselines.simple_raptor.retriever import SimpleRaptorRetriever
        from baselines.naive_rag.runner import LLMClient
        from config.config_loader import config as global_config_loader

        # Default paths or from indexes-dir
        index_path = "indexes/simple_raptor_index.faiss"
        nodes_path = "indexes/simple_raptor_nodes.pkl"
        chunk_path = "indexes/simple_raptor_chunk_store.pkl"

        if args.indexes_dir:
             idx_root = Path(args.indexes_dir)
             if (idx_root / "simple_raptor_index.faiss").exists():
                 index_path = str(idx_root / "simple_raptor_index.faiss")
                 nodes_path = str(idx_root / "simple_raptor_nodes.pkl")
                 chunk_path = str(idx_root / "simple_raptor_chunk_store.pkl")

        logger.info(f"Running Simple Raptor with index={index_path}")

        # Create custom LLM Client if args provided
        llm_client = None
        if lm_endpoint and lm_model:
            llm_client = LLMClient(
                endpoint=lm_endpoint,
                model=lm_model,
                temperature=args.temperature if args.temperature is not None else 0.0,
                max_tokens=args.max_new_tokens or 2048
            )

        # Init retriever
        retriever = SimpleRaptorRetriever(
            index_path=index_path,
            nodes_path=nodes_path,
            chunk_store_path=chunk_path,
            config=global_config_loader.load_config(),
            llm_client=llm_client
        )

        results = []
        qa_lines = []

        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            
            try:
                ans = retriever.answer(question)
                # Simple cleaning
                if ans.lower().startswith("answer:"):
                    ans = ans[7:].strip()
                clean_ans = _strip_reasoning(ans)
                clean_ans_line = " ".join(clean_ans.split())
                
                results.append({
                    "query_id": qid,
                    "question": question,
                    "answer": clean_ans,
                    "raw_answer": ans
                })
                qa_lines.append(f"{question}\t{clean_ans_line}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        
        if qa_log_path:
            qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            qa_log_path.write_text("\n".join(qa_lines), encoding="utf-8")
            
        logger.info("Simple Raptor baseline complete.")
        return

    indexes_dir = Path(args.indexes_dir) if args.indexes_dir else artifacts_dir / "indexes"
    notes_path = Path(args.notes) if args.notes else artifacts_dir / "notes" / f"notes.{dataset_name}.jsonl"
    if not indexes_dir.exists():
        raise FileNotFoundError(f"Indexes dir not found: {indexes_dir}")
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}")

    from query.query_processor import QueryProcessor

    qp = QueryProcessor(
        indexes_dir=str(indexes_dir),
        notes_path=str(notes_path),
    )

    results = []
    qa_lines: List[str] = []
    for item in dataset:
        question = item.get("query") or item.get("question")
        if not question:
            continue
        attr_hint = "occupation" if "occupation" in question.lower() else None
        qid = item.get("query_id")
        doc_hint = None
        if qid:
            doc_hint = f"{dataset_name}/{qid}"
        res = qp.process(question, doc_hint=doc_hint, attribute_hint=attr_hint)
        results.append(
            {
                "query_id": item.get("query_id"),
                "question": question,
                "answer": res.get("answer"),
                "structured": res.get("structured"),
                "decision": res.get("decision"),
            }
        )
        answer_text = res.get("answer")
        if answer_text is None:
            clean_answer = ""
        else:
            stripped = _strip_reasoning(str(answer_text))
            clean_answer = " ".join(stripped.splitlines())
        qa_lines.append(f"{question}\t{clean_answer}")
        
        # Log retrieval for evaluation
        try:
            structured = res.get("structured") or {}
            evidences = structured.get("evidence") or []
            # Convert evidences to format expected by log_retrieval
            retrieved_items = []
            for idx, ev in enumerate(evidences):
                # Try to map note_id or source info to doc_id/passage_id if possible
                # Assuming note_id might be useful or if there's source info
                item = {
                    "rank": idx + 1,
                    "text": ev.get("evidence") or ev.get("text"),
                    "score": ev.get("score"),
                    "doc_id": ev.get("doc_id") or ev.get("note_id"),
                    "passage_id": ev.get("passage_id") or ev.get("note_id")
                }
                retrieved_items.append(item)
            
            log_retrieval(
                sample_id=qid or str(hash(question)),
                dataset=dataset_name,
                run_name=work_dir.name,
                retrieved=retrieved_items,
                log_dir=work_dir
            )
        except Exception as e:
            logger.warning(f"Failed to log retrieval for {qid}: {e}")

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
