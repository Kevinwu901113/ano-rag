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
        
    return dataset

def load_hotpotqa_distractor(path: Path) -> List[dict]:
    """
    Load HotpotQA distractor dataset which includes full context.
    Structure: id, question, answer, context, supporting_facts
    """
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA dataset not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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
    parser.add_argument("--lm-endpoint", default=None)
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--qa-log", default=None, help="Optional plain text QA log path (question \t answer)")
    parser.add_argument("--hotpot-eval-output", default=None, help="Path to output official HotpotQA prediction JSON")
    parser.add_argument("--include-gold-sp", action="store_true", help="Include gold supporting facts in prediction file (upper bound for SP)")
    parser.add_argument("--new", action="store_true", help="Force create a new workspace copy")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override LLM max new tokens")
    args = parser.parse_args()

    lm_endpoint = args.lm_endpoint or global_config_loader.get("vllm.endpoint")
    lm_model = args.lm_model or global_config_loader.get("vllm.model")

    # Normalize baseline selection
    if args.direct_llm: args.baseline = "direct"
    if args.vanilla_rag: args.baseline = "vanilla"
    if args.simple_raptor: args.baseline = "raptor"

    if args.dataset == "hotpotqa":
        dataset_path = args.dataset_path if args.dataset_path else "data/hotpotqa/dataset_distractor.json"
        dataset = load_hotpotqa_distractor(Path(dataset_path))
        # Ensure consistency by preferring _id if available (official format)
        for item in dataset:
            if "_id" not in item and "id" in item:
                item["_id"] = item["id"]
    else:
        dataset = load_dataset_file(args.dataset, args.dataset_path)

    result_root = Path(args.result_root)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Include baseline in workspace name if not default
        ws_name = args.dataset
        if args.baseline != "mirage":
             ws_name = f"{args.dataset}_{args.baseline}"
        
        result_root.mkdir(parents=True, exist_ok=True)
        work_dir = _select_workspace(result_root, ws_name, args.new)
    logger.info("Using workspace: {}", work_dir)

    output_path = Path(args.out) if args.out else work_dir / "answers.json"
    qa_log_path = Path(args.qa_log) if args.qa_log else work_dir / "qa.tsv"

    if args.limit > 0:
        dataset = dataset[: args.limit]

    # Setup LLM Client
    llm_client = LLMClient(
        endpoint=lm_endpoint,
        model=lm_model,
        temperature=args.temperature if args.temperature is not None else 0.0,
        max_tokens=args.max_new_tokens or 2048
    )

    results = []
    qa_lines = []

    # --- BASELINE RUNNERS ---

    if args.baseline == "direct":
        # Handle HotpotQA distractor setting for direct/naive baseline by constructing context from paragraphs
        if args.dataset == "hotpotqa":
             # For HotpotQA, we construct context from the 10 distractor paragraphs
            for i, item in enumerate(dataset):
                question = item.get("question")
                qid = item.get("id") or str(i)
                
                # Construct context from title + sentences
                ctx_titles = item["context"]["title"]
                ctx_sents = item["context"]["sentences"]
                
                paragraphs = []
                for title, sents in zip(ctx_titles, ctx_sents):
                    para_text = f"Title: {title}\n" + " ".join(sents)
                    paragraphs.append(para_text)
                
                context_block = "\n\n".join(paragraphs)
                
                prompt = f"""You are answering a multi-hop question based on the provided documents.
    
Question:
{question}

Context:
{context_block}

Answer with a short phrase. If the answer is not in the context, say "unknown"."""
                
                try:
                    ans = llm_client.chat([
                        {"role": "user", "content": prompt}
                    ])
                    clean_ans = _strip_reasoning(ans)
                    # Prefer _id for HotpotQA
                    res_id = item.get("_id") or item.get("id") or str(i)
                    results.append({"query_id": res_id, "question": question, "answer": clean_ans, "raw_answer": ans})
                    qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
                except Exception as e:
                    logger.error(f"Error Q{i}: {e}")

        else:
            runner = DirectLLMRunner(
                lm_endpoint=lm_endpoint,
                lm_model=lm_model,
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
            
            # DirectLLMRunner produces "answers_jsonl" which has raw records. We need to convert to Hotpot eval format if requested
            if args.hotpot_eval_output:
                if args.dataset != "hotpotqa":
                    logger.warning("Using --hotpot-eval-output with a non-HotpotQA dataset. Ensure this is intended.")

                from collections import OrderedDict
                pred_answers = OrderedDict()
                # Load results
                with open(artifacts["answers_json"], "r", encoding="utf-8") as f:
                    final_res = json.load(f)
                for item in final_res:
                    qid = item.get("query_id") or item.get("id")
                    ans = item.get("answer") or ""
                    pred_answers[qid] = ans
                
                eval_out = {
                    "answer": pred_answers,
                    "sp": {qid: [] for qid in pred_answers.keys()}
                }
                out_p = Path(args.hotpot_eval_output)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(eval_out, f, ensure_ascii=False)
                logger.info("Saved HotpotQA evaluation prediction to {}", out_p)
                
            return

    elif args.baseline == "naive":
        if args.dataset == "hotpotqa":
             # Re-use the same logic as direct for HotpotQA distractor (as per instructions: "10 paragraphs all fed in")
             # Essentially treating naive/vanilla/etc as "RAG over 10 docs" or just "Long Context"
             # Since the instruction says: "All baselines use '10 paragraphs fed in + different prompt/structure'"
             # But simplest start is just feed them all.
             
            for i, item in enumerate(dataset):
                question = item.get("question")
                # Prefer _id for HotpotQA
                qid = item.get("_id") or item.get("id") or str(i)
                
                ctx_titles = item["context"]["title"]
                ctx_sents = item["context"]["sentences"]
                
                paragraphs = []
                for title, sents in zip(ctx_titles, ctx_sents):
                    para_text = f"Title: {title}\n" + " ".join(sents)
                    paragraphs.append(para_text)
                
                context_block = "\n\n".join(paragraphs)
                
                prompt = f"""You are answering a multi-hop question.
    
Question:
{question}

Context:
{context_block}

Answer with a short phrase. If the answer is not in the context, say "unknown"."""

                try:
                    ans = llm_client.chat([
                        {"role": "user", "content": prompt}
                    ])
                    clean_ans = _strip_reasoning(ans)
                    # Prefer _id for HotpotQA
                    res_id = item.get("_id") or item.get("id") or str(i)
                    results.append({"query_id": res_id, "question": question, "answer": clean_ans, "raw_answer": ans})
                    qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
                except Exception as e:
                    logger.error(f"Error Q{i}: {e}")
        else:
            from baselines.naive_rag.runner import NaiveRAGRunner
            # Infer index path
            index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{dataset_name}_naive")
            meta_path = index_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    index_path = str(Path(str(meta.get("index") or "")).expanduser())
                    chunk_path = str(Path(str(meta.get("chunks") or "")).expanduser())
                except Exception:
                    index_path = str(index_dir / "index.faiss")
                    chunk_path = str(index_dir / "chunks.jsonl")
            else:
                index_path = str(index_dir / "index.faiss")
                chunk_path = str(index_dir / "chunks.jsonl")

            runner = NaiveRAGRunner(
                index_path=index_path,
                chunks_path=chunk_path,
                lm_endpoint=lm_endpoint,
                lm_model=lm_model,
            )
            answer_func = runner.answer
            
            for i, item in enumerate(dataset):
                question = item.get("query") or item.get("question")
                qid = item.get("query_id") or str(i)
                try:
                    ans = answer_func(question)
                    clean_ans = _strip_reasoning(ans)
                    results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                    qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
                except Exception as e:
                    logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "relrag":
        from baselines.simple_graphrag.runner import answer as relrag_answer
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{args.dataset}_relrag")
        logger.info(f"Running RelRAG with index_dir={index_dir}")

        for i, item in enumerate(dataset):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            try:
                ans = relrag_answer(question, index_dir=str(index_dir))
                clean_ans = _strip_reasoning(ans)
                results.append({"query_id": qid, "question": question, "answer": clean_ans, "raw_answer": ans})
                qa_lines.append(f"{question}\t{' '.join(clean_ans.split())}")
            except Exception as e:
                logger.error(f"Error Q{i}: {e}")

    elif args.baseline == "vanilla":
        from baselines.vanilla_rag import answer as vanilla_rag_answer
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{args.dataset}_vanilla")
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
                global_config_loader.set("vllm.endpoint", lm_endpoint)
                global_config_loader.set("vllm.model", lm_model)
                
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
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{args.dataset}_selfrag")
        index_path = str(index_dir / "selfrag_index.faiss")
        chunk_path = str(index_dir / "selfrag_chunk_store.pkl")
        
        logger.info(f"Running Self-RAG with index={index_path}")
        
        # Set global config for LLM
        global_config_loader.set("vllm.endpoint", lm_endpoint)
        global_config_loader.set("vllm.model", lm_model)

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
        
        index_dir = Path(args.indexes_dir) if args.indexes_dir else Path(f"result/{args.dataset}_raptor")
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
        global_config_loader.set("vllm.endpoint", lm_endpoint)
        global_config_loader.set("vllm.model", lm_model)
        
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

    # --- Save results ---
    if results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Saved JSON results to {}", output_path)

        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(qa_lines))
        logger.info("Saved QA log to {}", qa_log_path)
        
    if args.hotpot_eval_output:
        if args.dataset != "hotpotqa":
            logger.warning("Using --hotpot-eval-output with a non-HotpotQA dataset. Ensure this is intended.")
        
        from collections import OrderedDict
        pred_answers = OrderedDict()
        pred_sp = OrderedDict()
        
        # Build a lookup for gold SP if requested
        gold_sp_map = {}
        if args.include_gold_sp:
            for item in dataset:
                qid = item.get("_id") or item.get("id")
                qid = str(qid)
                # Format: [ [title, sent_id], ... ]
                # In HF dataset, it is {"title": [...], "sent_id": [...]}
                # In official json, it is [ [title, sent_id], ... ]
                # We need to adapt based on source format
                sp_raw = item.get("supporting_facts")
                sp_list = []
                if isinstance(sp_raw, dict): # HF format
                     titles = sp_raw.get("title", [])
                     sent_ids = sp_raw.get("sent_id", [])
                     for t, s in zip(titles, sent_ids):
                         sp_list.append([t, s])
                elif isinstance(sp_raw, list): # Official format
                     sp_list = sp_raw
                gold_sp_map[qid] = sp_list

        for item in results:
            # For HotpotQA, qid is now unified to _id (which is same as id)
            # For other datasets, it might be query_id or id.
            # We trust 'query_id' field in results list which we populated above.
            qid = item.get("query_id") or item.get("id")
            qid = str(qid)
            ans = item.get("answer") or ""
            pred_answers[qid] = ans
            pred_sp[qid] = gold_sp_map.get(qid, []) if args.include_gold_sp else []
        
        eval_out = {
            "answer": pred_answers,
            "sp": pred_sp
        }
        out_p = Path(args.hotpot_eval_output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(eval_out, f, ensure_ascii=False)
        logger.info("Saved HotpotQA evaluation prediction to {}", out_p)

    logger.info("Done.")


if __name__ == "__main__":
    main()
