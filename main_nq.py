#!/usr/bin/env python3
"""Natural Questions (NQ) 批量预测入口。"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Any, List

from answer.nq_answer import decide_nq_answer
from answer.efsa_answer import EfsaAnswer
from answer.verify_shell import Verifier
from retrieval.quick_hybrid_test import HybridRetriever


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _support_contexts(support: List[Dict[str, Any]]) -> List[str]:
    contexts: List[str] = []
    for entry in support:
        text = entry.get("text") if isinstance(entry, dict) else None
        if text:
            contexts.append(text)
    return contexts


def process_item(
    item: Dict[str, Any],
    retriever: HybridRetriever,
    efsa: EfsaAnswer,
    verifier: Verifier,
    topk: int = 20,
) -> Dict[str, Any]:
    qid = str(item.get("id", ""))
    question = item.get("question", "")
    paragraphs = item.get("paragraphs", [])

    ranked = retriever.rank(question, paragraphs) or []
    retrieved_doc_ids: List[str] = []
    for r in ranked[:topk]:
        try:
            retrieved_doc_ids.append(f"{qid}_{int(r['idx'])}")
        except (KeyError, TypeError, ValueError):
            continue

    support = ranked[: min(3, len(ranked))]
    predicted_support_idxs: List[int] = []
    for s in support:
        try:
            predicted_support_idxs.append(int(s["idx"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not predicted_support_idxs and support:
        predicted_support_idxs = list(range(len(support)))

    efsa_candidate, conf_scores, support_sents = efsa.produce(
        question=question,
        contexts=_support_contexts(support),
    )

    predicted_answer, predicted_answerable = decide_nq_answer(
        example=item,
        efsa_candidate=efsa_candidate,
        support_sents=support_sents,
        conf_scores=conf_scores or {},
        verifier=verifier,
    )

    return {
        "id": qid,
        "predicted_answer": predicted_answer,
        "predicted_answerable": bool(predicted_answerable),
        "predicted_support_idxs": predicted_support_idxs,
        "retrieved_doc_ids": retrieved_doc_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NQ pipeline on musique-style input")
    ap.add_argument("input_file", help="data/nq_dev/musique_style_dev.jsonl")
    ap.add_argument("output_file", help="result/nq_dev/nq_results.jsonl")
    ap.add_argument("--workers", type=int, default=4, help="并行工作线程数（当前实现为串行，仅用于接口兼容）")
    ap.add_argument("--topk", type=int, default=20, help="检索返回的段落数量")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    logger.info("loading dataset from %s", args.input_file)
    data = _load_dataset(args.input_file)
    logger.info("loaded %d examples", len(data))

    retriever = HybridRetriever(top_k=args.topk)
    efsa = EfsaAnswer()
    verifier = Verifier()

    results: List[Dict[str, Any]] = []
    for item in data:
        try:
            results.append(process_item(item, retriever, efsa, verifier, topk=args.topk))
        except Exception as exc:  # pragma: no cover - 兜底，防止批量中断
            logger.exception("failed to process item %s", item.get("id"))
            results.append(
                {
                    "id": str(item.get("id")),
                    "error": str(exc),
                    "predicted_answer": "",
                    "predicted_answerable": False,
                    "predicted_support_idxs": [],
                    "retrieved_doc_ids": [],
                }
            )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[OK] wrote %s ; total=%d", args.output_file, len(results))


if __name__ == "__main__":
    main()
