#!/usr/bin/env python3
"""
Diagnostic runner for the HotpotQA RelRAG baseline.

Usage (defaults match baseline settings):
    python scripts/hotpotqa/baselines/run_relrag_diagnostic.py \
        --dataset data/hotpotqa/dataset_distractor_200.json \
        --lm-endpoint http://127.0.0.1:1234/v1 \
        --lm-model qwen/qwen3-30b-a3b \
        --output-dir analysis/hotpotqa_relrag_case_study_20
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (  # noqa: E402
    build_passages_from_context,
    detect_device,
    get_embedding_model,
)
from utils.answer_cleaner import _strip_reasoning, clean_model_answer


def clean_hotpot_answer(text: str) -> str:
    """
    Strip <think>...</think> and common prefixes to keep only the final short answer.
    Provides fallback to avoid empty outputs.
    """
    if text is None:
        return ""
    raw = str(text)
    cleaned = clean_model_answer(raw)
    cleaned = _strip_reasoning(cleaned)
    cleaned = re.sub(r"^answer\s*[:\uff1a]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("\"'\u201c\u201d\u2018\u2019").strip()
    cleaned = " ".join(cleaned.split())
    if cleaned:
        return cleaned
    fallback = re.sub(r"^answer\s*[:\uff1a]\s*", "", raw, flags=re.IGNORECASE).strip()
    fallback = fallback.strip("\"'\u201c\u201d\u2018\u2019").strip()
    fallback = " ".join(fallback.split())
    return fallback or "Insufficient evidence"


def normalize_answer(text: str) -> str:
    """HotpotQA official normalization (copied from hotpot_evaluate_v1)."""
    import re
    import string

    def remove_articles(s: str) -> str:
        return re.sub(r"\\b(a|an|the)\\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, gold: str) -> Tuple[float, float, float]:
    """Return (f1, precision, recall)."""
    from collections import Counter

    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_examples(data: List[Dict[str, Any]], sample_size: int, seed: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    selected = indices[:sample_size]
    subset = [data[i] for i in selected]
    return subset, selected


def _normalize(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norm + 1e-10)


def _build_passages_with_meta(context: Any, max_passages: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Return (passage_texts, meta) while keeping the exact passage strings used by the baseline.
    """
    passages = build_passages_from_context(context, max_passages=max_passages)
    meta: List[Dict[str, Any]] = []

    if isinstance(context, dict):
        titles = list(context.get("title", []))[:max_passages]
        sentences_list = list(context.get("sentences", []))[:max_passages]
        for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
            meta.append(
                {
                    "id": f"p{idx}",
                    "title": title,
                    "sentences": list(sentences),
                }
            )
    else:
        for idx, (title, sentences) in enumerate(list(context)[:max_passages]):
            meta.append(
                {
                    "id": f"p{idx}",
                    "title": title,
                    "sentences": list(sentences),
                }
            )
    return passages, meta


def _supporting_sentences(meta: List[Dict[str, Any]], supporting_facts: Any) -> List[Dict[str, Any]]:
    """
    Normalize supporting facts (HotpotQA distractor variant stores as dict of lists).
    """
    title_to_idx = {m["title"]: idx for idx, m in enumerate(meta)}
    resolved: List[Dict[str, Any]] = []

    pairs: List[Tuple[Any, Any]] = []
    if isinstance(supporting_facts, dict):
        titles = list(supporting_facts.get("title") or [])
        sent_ids = list(supporting_facts.get("sent_id") or supporting_facts.get("sent_ids") or [])
        for t, sid in zip(titles, sent_ids):
            pairs.append((t, sid))
    else:
        for fact in supporting_facts or []:
            if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                pairs.append((fact[0], fact[1]))

    for title, sent_idx in pairs:
        entry: Dict[str, Any] = {"title": title, "sentence_index": sent_idx}
        if title in title_to_idx:
            pidx = title_to_idx[title]
            entry["passage_id"] = meta[pidx]["id"]
            sentences = meta[pidx].get("sentences", [])
            if 0 <= sent_idx < len(sentences):
                entry["text"] = sentences[sent_idx]
        resolved.append(entry)
    return resolved


class RelRAGDiagnosticRunner:
    """
    RelRAG with rich logging for per-sample artifact capture.
    Mirrors scripts/hotpotqa/baselines/run_relrag.py for scoring.
    """

    def __init__(
        self,
        encoder,
        llm: LLMChatClient,
        *,
        threshold: float = 0.7,
        alpha: float = 0.6,
        topk: int = 3,
        max_context: int = 10,
    ) -> None:
        self.encoder = encoder
        self.llm = llm
        self.threshold = threshold
        self.alpha = alpha
        self.topk = topk
        self.max_context = max_context

    def run_case(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("_id") or item.get("id") or ""
        question = str(item.get("question") or "")
        gold_answer = str(item.get("answer") or "")
        supporting_facts = item.get("supporting_facts", [])

        case: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "ground_truth": {
                "answer": gold_answer,
                "supporting_facts": supporting_facts,
            },
            "retrieval": {},
            "graph": {},
            "agent": {},
            "llm": {},
            "postprocess": {},
            "metrics": {},
            "timing_ms": {},
        }

        passages, meta = _build_passages_with_meta(item.get("context"), max_passages=self.max_context)
        case["graph"]["nodes"] = [{"id": m["id"], "title": m["title"]} for m in meta]
        case["ground_truth"]["supporting_sentences"] = _supporting_sentences(meta, supporting_facts)
        support_titles: List[str] = []
        if isinstance(supporting_facts, dict):
            support_titles = list(supporting_facts.get("title") or [])
        elif isinstance(supporting_facts, list):
            for fact in supporting_facts:
                if isinstance(fact, (list, tuple)) and len(fact) >= 1:
                    support_titles.append(fact[0])
        support_titles = [t for t in support_titles if isinstance(t, str)]

        if not passages:
            case["error"] = "no_passages_found"
            case["final_answer"] = "Insufficient evidence"
            return case

        run_start = time.perf_counter()
        try:
            enc_start = time.perf_counter()
            vecs = self.encoder(passages)
            enc_ms = (time.perf_counter() - enc_start) * 1000
            vecs = _normalize(vecs)

            sim_matrix = np.dot(vecs, vecs.T)
            adj = (sim_matrix > self.threshold).astype(float)

            q_vec = self.encoder([question])
            q_vec = _normalize(q_vec)

            initial_scores = np.dot(vecs, q_vec.T).flatten()
            neighbor_scores = np.dot(adj, initial_scores) / (np.sum(adj, axis=1) + 1e-10)
            final_scores = self.alpha * initial_scores + (1 - self.alpha) * neighbor_scores

            sorted_idx = list(np.argsort(final_scores)[::-1])
            selected_idx = sorted_idx[: self.topk]

            edges: List[Dict[str, Any]] = []
            for i in range(len(passages)):
                for j in range(len(passages)):
                    if adj[i, j] > 0:
                        edges.append(
                            {
                                "source": meta[i]["id"],
                                "target": meta[j]["id"],
                                "similarity": float(sim_matrix[i, j]),
                            }
                        )

            candidates: List[Dict[str, Any]] = []
            for i, passage in enumerate(passages):
                candidates.append(
                    {
                        "id": meta[i]["id"],
                        "title": meta[i]["title"],
                        "text": passage,
                        "question_score": float(initial_scores[i]),
                        "neighbor_score": float(neighbor_scores[i]),
                        "final_score": float(final_scores[i]),
                        "selected": i in selected_idx,
                    }
                )

            selected_ids = [meta[i]["id"] for i in selected_idx]
            selected_texts = [passages[i] for i in selected_idx]
            context_block = "\n\n".join(selected_texts)
            prompt = (
                "Answer the question based on the context.\n\n"
                f"{context_block}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
            messages = [{"role": "user", "content": prompt}]

            llm_start = time.perf_counter()
            llm_resp = self.llm.chat(messages)
            llm_ms = (time.perf_counter() - llm_start) * 1000

            raw_answer = llm_resp.content
            clean_answer = clean_hotpot_answer(raw_answer)
            f1, prec, recall = f1_score(clean_answer, gold_answer)
            em = exact_match(clean_answer, gold_answer)

            case["retrieval"] = {
                "query": question,
                "query_rewrites": [],
                "candidates": candidates,
                "similarity_threshold": self.threshold,
                "pruning": {
                    "kept_passage_ids": [m["id"] for m in meta],
                    "dropped_passage_ids": [],
                },
            }
            case["graph"]["edges"] = edges
            case["reranking"] = {
                "alpha": self.alpha,
                "topk": self.topk,
                "sorted_passage_ids": [meta[i]["id"] for i in sorted_idx],
                "selected_passage_ids": selected_ids,
            }
            case["agent"] = {
                "steps": [
                    {
                        "step_index": 1,
                        "action": "answer",
                        "context_passage_ids": selected_ids,
                        "prompt": prompt,
                    }
                ]
            }
            case["llm"] = {
                "endpoint": self.llm.endpoint,
                "model": self.llm.model,
                "messages": messages,
                "raw_response": llm_resp.raw,
                "response_text": raw_answer,
            }
            case["postprocess"] = {
                "answer": clean_answer,
                "cleaning": "clean_hotpot_answer",
            }
            case["final_answer"] = clean_answer
            case["metrics"] = {
                "em": bool(em),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(recall),
            }
            case["selected_context"] = [
                {"passage_id": meta[i]["id"], "title": meta[i]["title"], "text": passages[i]} for i in selected_idx
            ]
            selected_titles = [ctx["title"] for ctx in case["selected_context"]]
            matched_titles = [t for t in support_titles if t in selected_titles]
            case["coverage"] = {
                "support_title_total": len(support_titles),
                "selected_titles": selected_titles,
                "matched_titles": matched_titles,
            }
            case["timing_ms"] = {
                "embedding": enc_ms,
                "llm": llm_ms,
                "total": (time.perf_counter() - run_start) * 1000,
            }
            return case
        except Exception as exc:  # noqa: PERF203
            logger.exception("Failed to process example %s", qid)
            case["error"] = str(exc)
            case["final_answer"] = "error"
            case["metrics"] = None
            return case


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_config(
    path: Path,
    *,
    dataset_path: Path,
    seed: int,
    sample_size: int,
    selected_indices: List[int],
    selected_ids: List[str],
    args: argparse.Namespace,
    device: str,
) -> None:
    config = {
        "dataset_path": str(dataset_path),
        "seed": seed,
        "sample_size": sample_size,
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
        "retrieval": {
            "threshold": args.threshold,
            "alpha": args.alpha,
            "topk": args.topk,
            "max_context": args.max_context,
        },
        "llm": {
            "endpoint": args.lm_endpoint,
            "model": args.lm_model,
            "temperature": args.temperature,
        },
        "embedding": {
            "model": args.emb_model,
            "device": device,
            "dtype": args.emb_dtype or "auto",
        },
        "runner": {
            "script": Path(__file__).name,
            "command": " ".join(sys.argv),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_summary(path: Path, cases: List[Dict[str, Any]]) -> None:
    valid = [c for c in cases if c.get("metrics")]
    errors = [c for c in cases if not c.get("metrics")]
    total = len(cases)
    if valid:
        em = sum(1.0 if c["metrics"]["em"] else 0.0 for c in valid) / len(valid)
        f1 = sum(c["metrics"]["f1"] for c in valid) / len(valid)
        prec = sum(c["metrics"]["precision"] for c in valid) / len(valid)
        recall = sum(c["metrics"]["recall"] for c in valid) / len(valid)
    else:
        em = f1 = prec = recall = 0.0

    full_cov = 0
    partial_cov = 0
    zero_cov = 0
    for c in cases:
        cov = c.get("coverage") or {}
        tot = int(cov.get("support_title_total") or 0)
        matched = len(cov.get("matched_titles") or [])
        if tot == 0:
            continue
        if matched >= tot and tot > 0:
            full_cov += 1
        elif matched > 0:
            partial_cov += 1
        else:
            zero_cov += 1

    lines = [
        f"Total cases: {total}",
        f"Completed (with metrics): {len(valid)}",
        f"Errors: {len(errors)}",
        f"EM: {em:.4f}",
        f"F1: {f1:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall: {recall:.4f}",
        f"Support coverage (titles) - full: {full_cov}, partial: {partial_cov}, zero: {zero_cov}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RelRAG on a sampled subset with full artifacts.")
    parser.add_argument("--dataset", default="data/hotpotqa/dataset_distractor_200.json")
    parser.add_argument("--output-dir", default="analysis/hotpotqa_relrag_case_study_20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--lm-model", default="qwen/qwen3-30b-a3b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--emb-device", default=None)
    parser.add_argument("--emb-dtype", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    data = load_dataset(str(dataset_path))
    subset, selected_indices = sample_examples(data, args.sample_size, args.seed)
    logger.info("Sampled %d / %d examples (seed=%s)", len(subset), len(data), args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.jsonl"
    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.txt"
    pred_path = out_dir / "pred.json"

    device = args.emb_device or detect_device()
    encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=args.temperature,
    )
    runner = RelRAGDiagnosticRunner(
        encoder,
        llm,
        threshold=args.threshold,
        alpha=args.alpha,
        topk=args.topk,
        max_context=args.max_context,
    )

    cases: List[Dict[str, Any]] = []
    predictions = {"answer": {}, "sp": {}}

    for item in tqdm(subset, desc="Running RelRAG"):
        case = runner.run_case(item)
        cases.append(case)
        qid = case.get("id")
        if qid:
            predictions["answer"][qid] = case.get("final_answer", "")
            predictions["sp"][qid] = []

    write_jsonl(cases_path, cases)
    write_config(
        config_path,
        dataset_path=dataset_path.resolve(),
        seed=args.seed,
        sample_size=args.sample_size,
        selected_indices=selected_indices,
        selected_ids=[c.get("id") for c in cases],
        args=args,
        device=device,
    )
    write_summary(summary_path, cases)
    pred_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved cases to %s", cases_path)
    logger.info("Saved config to %s", config_path)
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved predictions to %s", pred_path)


if __name__ == "__main__":
    main()
