#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import DEFAULT_EMBED_DEVICE, DEFAULT_EMBED_MODEL, config as global_config
from scripts.hotpotqa.baselines.baseline_utils import TransformerEmbedder
from utils.context_budget import pack_contexts
from utils.device import run_with_fallback
from utils.jsonl_utils import write_jsonl
from utils.llm_client import LLMChatClient
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format: {type(data)}")


def _load_doc_pool(doc_pool_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(doc_pool_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if isinstance(value, dict):
                items.append({**value, "doc_id": value.get("doc_id") or key})
            else:
                items.append({"doc_id": key, "doc_chunk": value})
        return items
    raise ValueError(f"Unsupported doc pool format: {type(data)}")


def _build_entries(doc_pool: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, item in enumerate(doc_pool):
        text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
        text = str(text or "").strip()
        if not text:
            continue
        doc_name = str(item.get("doc_name") or item.get("title") or "").strip()
        # MIRAGE canonical id must be UUID (mapped_id/query_id). DO NOT use title as doc_id.
        canonical_id = item.get("mapped_id") or item.get("doc_id")
        doc_id = str(canonical_id or doc_name or f"doc_{idx:06d}")
        full_text = f"Title: {doc_name}\nContent: {text}" if doc_name else text
        entries.append(
            {
                "text": full_text,
                "title": doc_name,
                "doc_id": doc_id,
                "sent_ids": None,
                "passage_id": f"{doc_id}::{idx}",
            }
        )
    return entries


def _load_resume_state(
    output_path: Path,
    qa_path: Path,
    pred_raw_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], set[str], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []
    completed: set[str] = set()

    if pred_raw_path.exists():
        try:
            for line in pred_raw_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pred_raw_records.append(row)
                sid = str(row.get("id") or row.get("query_id") or row.get("_id") or "")
                if sid:
                    completed.add(sid)
        except Exception as exc:
            logger.warning("Failed to load existing pred_raw.jsonl: {}", exc)

    if output_path.exists():
        try:
            for line in output_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed to load existing results from {}: {}", output_path, exc)

    if qa_path.exists():
        try:
            for line in qa_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if "\t" in line:
                    q, a = line.split("\t", 1)
                else:
                    q, a = line, ""
                qa_rows.append((q, a))
        except Exception as exc:
            logger.warning("Failed to load existing QA log from {}: {}", qa_path, exc)

    return results, qa_rows, completed, pred_raw_records


class RelRAGIndex:
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        embedder: TransformerEmbedder,
        *,
        embed_device: str,
        sim_threshold: float = 0.7,
        hop: int = 1,
        alpha: float = 0.6,
        use_scheduler: bool = True,
    ) -> None:
        if not entries:
            raise ValueError("No document entries available for retrieval.")
        self.entries = entries
        self.embedder = embedder
        self.embed_device = embed_device
        self.sim_threshold = float(sim_threshold)
        self.hop = max(1, int(hop))
        self.alpha = float(alpha)
        self.use_scheduler = bool(use_scheduler)
        self.embed_used_devices: set[str] = set()
        self.embed_fallback_reasons: List[str] = []

        texts = [e["text"] for e in entries]
        vecs, used_device, fallback_reason = run_with_fallback(
            lambda device: self.embedder.encode(texts, device=device),
            prefer=self.embed_device,
        )
        self.embed_used_devices.add(str(used_device))
        if fallback_reason:
            self.embed_fallback_reasons.append(str(fallback_reason))
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.size == 0:
            raise RuntimeError("Embedding encoder returned empty vectors.")
        vecs = self._normalize(vecs)
        self.vectors = vecs
        self.adj = None
        if self.use_scheduler:
            self.adj = self._build_adjacency(vecs)

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norm + 1e-10)

    def _build_adjacency(self, vecs: np.ndarray) -> np.ndarray:
        sim = vecs @ vecs.T
        adj = (sim > self.sim_threshold).astype("float32")
        if self.hop > 1:
            adj_hop = adj.copy()
            for _ in range(1, self.hop):
                adj_hop = (adj_hop @ adj) > 0
            adj = adj_hop.astype("float32")
        return adj

    def retrieve(
        self,
        question: str,
        *,
        topk: int,
        context_budget: int,
    ) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], int]:
        q_vec, used_device, fallback_reason = run_with_fallback(
            lambda device: self.embedder.encode([question], device=device),
            prefer=self.embed_device,
        )
        self.embed_used_devices.add(str(used_device))
        if fallback_reason:
            self.embed_fallback_reasons.append(str(fallback_reason))
        q_vec = np.asarray(q_vec, dtype="float32")
        if q_vec.size == 0:
            return [], "", [], 0
        q_vec = self._normalize(q_vec)

        initial_scores = (self.vectors @ q_vec.T).reshape(-1)
        final_scores = initial_scores
        neighbor_scores = None
        if self.use_scheduler and self.adj is not None:
            denom = self.adj.sum(axis=1) + 1e-10
            neighbor_scores = (self.adj @ initial_scores) / denom
            alpha = min(max(self.alpha, 0.0), 1.0)
            final_scores = alpha * initial_scores + (1.0 - alpha) * neighbor_scores

        limit = max(1, int(topk))
        indices = np.argsort(final_scores)[::-1][:limit]

        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            entry = self.entries[int(idx)]
            hits.append(
                {
                    "text": entry.get("text"),
                    "title": entry.get("title"),
                    "score": float(final_scores[int(idx)]),
                    "initial_score": float(initial_scores[int(idx)]),
                    "neighbor_score": float(neighbor_scores[int(idx)]) if neighbor_scores is not None else 0.0,
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                    "rank": rank + 1,
                }
            )

        annotated_hits = [
            {**hit, "text": f"[{hit['rank']}] {hit.get('text', '')}"} for hit in hits
        ]
        context_str, contexts_used, context_tokens = pack_contexts(
            annotated_hits, int(context_budget or 0)
        )
        return hits, context_str, contexts_used, context_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RelRAG baseline on MIRAGE")
    parser.add_argument("--dataset", "--dataset-path", dest="dataset_path", required=True)
    parser.add_argument("--doc-pool", "--doc-pool-path", dest="doc_pool_path", default=None)
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--lm-model", default="qwen3-30b-a3b")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sim-threshold", type=float, default=0.7)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument(
        "--embed-model",
        "--emb-model",
        dest="embed_model",
        default=DEFAULT_EMBED_MODEL,
    )
    parser.add_argument(
        "--embed-device",
        "--emb-device",
        dest="embed_device",
        choices=["auto", "cuda", "cpu"],
        default=DEFAULT_EMBED_DEVICE,
    )
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--embed-max-length", type=int, default=512)
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true")
    norm.add_argument("--no-embed-normalize", dest="embed_normalize", action="store_false")
    parser.set_defaults(embed_normalize=True)
    parser.add_argument("--emb-dtype", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    doc_pool_path = Path(args.doc_pool_path) if args.doc_pool_path else dataset_path.parent / "doc_pool.json"
    if not doc_pool_path.exists():
        raise FileNotFoundError(f"Doc pool not found: {doc_pool_path}")

    dataset = _load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]
    logger.info("Loaded {} examples from {}", len(dataset), dataset_path)

    doc_pool = _load_doc_pool(doc_pool_path)
    entries = _build_entries(doc_pool)
    if not entries:
        raise RuntimeError("Doc pool yielded zero usable entries.")
    logger.info("Loaded {} doc pool entries from {}", len(entries), doc_pool_path)

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    dataset_name = "mirage"
    output_path = preds_dir / "results.jsonl"
    qa_path = preds_dir / "qa.tsv"
    pred_raw_path = preds_dir / "pred_raw.jsonl"
    setup_logging(str(work_dir / "run.log"))
    logger.info("Writing outputs to {}", work_dir)

    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            decode={
                "temperature": 0.0,
                "top_p": None,
                "repetition_penalty": None,
                "max_tokens": args.max_new_tokens,
            },
            embedding={
                "model": args.embed_model,
                "device": args.embed_device,
                "batch_size": args.embed_batch_size,
                "max_length": args.embed_max_length,
                "normalize": args.embed_normalize,
                "dtype": args.emb_dtype,
            },
            budgets={
                "context_budget_tokens": args.context_budget or None,
                "topk": args.topk,
            },
            extra={
                "sim_threshold": args.sim_threshold,
                "graph_hop": args.hop,
                "scheduler_enabled": not args.no_scheduler,
                "alpha": args.alpha,
            },
        ),
    )

    embedder = TransformerEmbedder(
        args.embed_model,
        torch_dtype=args.emb_dtype,
        batch_size=args.embed_batch_size,
        max_length=args.embed_max_length,
        normalize=args.embed_normalize,
    )

    if args.num_workers and int(args.num_workers) > 1:
        logger.info("num_workers={} requested; running sequentially for now.", args.num_workers)

    index = RelRAGIndex(
        entries,
        embedder,
        embed_device=args.embed_device,
        sim_threshold=args.sim_threshold,
        hop=args.hop,
        alpha=args.alpha,
        use_scheduler=not args.no_scheduler,
    )

    llm = None
    if not args.retrieval_only:
        llm = LLMChatClient(
            endpoint=args.lm_endpoint,
            model=args.lm_model,
            temperature=0.0,
            max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 1024,
        )

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume:
        results, qa_rows, completed, pred_raw_records = _load_resume_state(
            output_path, qa_path, pred_raw_path
        )
        if completed:
            logger.info("Resuming with {} existing predictions", len(completed))

    save_every = max(0, int(args.save_every))
    for i, item in enumerate(dataset):
        qid = str(item.get("query_id") or item.get("id") or "")
        if not qid or qid in completed:
            continue
        question = str(item.get("query") or item.get("question") or "").strip()
        if not question:
            continue
        try:
            logger.info("Processing Q{}: {}", i, question)
            hits, context_str, contexts_used, context_tokens = index.retrieve(
                question,
                topk=args.topk,
                context_budget=int(args.context_budget or 0),
            )

            if args.retrieval_only or llm is None:
                ans = ""
            else:
                prompt = f"""Answer the question based on the context.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""
                resp = llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 1024,
                )
                ans = resp.content

            results.append(
                {
                    "query_id": qid,
                    "question": question,
                    "answer": ans,
                    "raw_answer": ans,
                }
            )
            qa_rows.append((question, ans))
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
                logger.error("retrieval logging failed for {}: {}", qid, log_exc)
            completed.add(str(qid))

            if save_every and len(completed) % save_every == 0:
                write_jsonl(output_path, results)
                write_jsonl(pred_raw_path, pred_raw_records)
                qa_path.write_text(
                    "\n".join([f"{q}\t{a}" for q, a in qa_rows]), encoding="utf-8"
                )
        except Exception as exc:
            logger.error("Error processing {}: {}", qid or i, exc)

    write_jsonl(output_path, results)
    write_jsonl(pred_raw_path, pred_raw_records)
    qa_path.write_text(
        "\n".join([f"{q}\t{a}" for q, a in qa_rows]), encoding="utf-8"
    )
    logger.info("Finished. Results saved to {}", work_dir)


if __name__ == "__main__":
    main()
