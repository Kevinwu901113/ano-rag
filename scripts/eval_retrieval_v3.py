#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_gold_titles(row: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    supporting_facts = row.get("supporting_facts")
    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and item:
                title = _norm_text(item[0])
                if title:
                    titles.append(title)
    paragraphs = row.get("paragraphs")
    if isinstance(paragraphs, list):
        for item in paragraphs:
            if isinstance(item, dict) and item.get("is_supporting") is True:
                title = _norm_text(item.get("title"))
                if title:
                    titles.append(title)
    docs = row.get("docs")
    if isinstance(docs, list):
        for item in docs:
            if isinstance(item, dict) and item.get("is_supporting") is True:
                title = _norm_text(item.get("title"))
                if title:
                    titles.append(title)

    seen: set[str] = set()
    dedup: List[str] = []
    for title in titles:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(title)
    return dedup


def _extract_contexts(row: Dict[str, Any]) -> Any:
    for key in ("ctxs", "retrieved_context_topk", "retrieved_context_raw", "retrieved_context"):
        if key in row:
            return row.get(key)
    return []


def _classify_contexts(contexts: Any) -> str:
    if not isinstance(contexts, list) or not contexts:
        return "invalid"
    has_dict = any(isinstance(item, dict) for item in contexts)
    has_str = any(isinstance(item, str) for item in contexts)
    if has_dict and not has_str:
        return "structured_ranked"
    if has_str and not has_dict:
        return "text_only"
    return "invalid"


def _extract_titles_from_structured(contexts: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    for item in contexts:
        if not isinstance(item, dict):
            continue
        title = _norm_text(item.get("title") or item.get("doc_title") or item.get("source_title"))
        if title:
            titles.append(title)
    return titles


def _recall_at_k(retrieved_titles: List[str], gold_titles: List[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    gold_lower = {item.lower() for item in gold_titles}
    retrieved_lower = {item.lower() for item in retrieved_titles[:k]}
    return len(retrieved_lower & gold_lower) / float(len(gold_lower))


def _ie_at_k(retrieved_titles: List[str], gold_titles: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_lower = {item.lower() for item in gold_titles}
    seen: set[str] = set()
    hits = 0
    for title in retrieved_titles[:k]:
        key = title.lower()
        if key in gold_lower and key not in seen:
            hits += 1
            seen.add(key)
    return hits / float(k)


def _ndcg_at_k(retrieved_titles: List[str], gold_titles: List[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    gold_lower = {item.lower() for item in gold_titles}
    seen: set[str] = set()
    dcg = 0.0
    for idx, title in enumerate(retrieved_titles[:k]):
        key = title.lower()
        rel = 1.0 if key in gold_lower and key not in seen else 0.0
        if rel > 0:
            seen.add(key)
        dcg += rel / math.log2(idx + 2)

    ideal_hits = min(len(gold_lower), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _text_recall(contexts: List[str], gold_titles: List[str]) -> float:
    if not gold_titles:
        return 0.0
    combined = " ".join(str(item or "") for item in contexts).lower()
    if not combined:
        return 0.0
    hits = sum(1 for title in gold_titles if title.lower() in combined)
    return hits / float(len(gold_titles))


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def evaluate_retrieval(
    *,
    pred_file: Path,
    gold_file: Path,
    expected_count: int,
) -> Dict[str, Any]:
    gold_rows = _load_jsonl(gold_file)
    pred_rows = _load_jsonl(pred_file)

    gold_map: Dict[str, List[str]] = {}
    for row in gold_rows:
        qid = _norm_text(row.get("_id") or row.get("id"))
        if not qid:
            continue
        gold_map[qid] = _extract_gold_titles(row)

    pred_map: Dict[str, Dict[str, Any]] = {}
    duplicate_pred_count = 0
    for row in pred_rows:
        qid = _norm_text(row.get("_id") or row.get("id"))
        if not qid:
            continue
        if qid in pred_map:
            duplicate_pred_count += 1
            continue
        pred_map[qid] = row

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())
    overlap_ids = sorted(gold_ids & pred_ids)
    id_overlap_count = len(overlap_ids)
    evaluated_count = id_overlap_count
    alignment_ok = evaluated_count == int(expected_count)

    mode_counts = {
        "structured_ranked": 0,
        "text_only": 0,
        "invalid": 0,
    }

    recall_2: List[float] = []
    recall_5: List[float] = []
    ie_2: List[float] = []
    ie_5: List[float] = []
    ndcg_2: List[float] = []
    ndcg_5: List[float] = []
    recall_text: List[float] = []

    for qid in overlap_ids:
        pred_row = pred_map[qid]
        gold_titles = gold_map[qid]
        contexts = _extract_contexts(pred_row)
        mode = _classify_contexts(contexts)
        mode_counts[mode] += 1

        if mode == "structured_ranked":
            titles = _extract_titles_from_structured(contexts)
            if not titles:
                mode_counts["structured_ranked"] -= 1
                mode_counts["invalid"] += 1
                continue
            recall_2.append(_recall_at_k(titles, gold_titles, 2))
            recall_5.append(_recall_at_k(titles, gold_titles, 5))
            ie_2.append(_ie_at_k(titles, gold_titles, 2))
            ie_5.append(_ie_at_k(titles, gold_titles, 5))
            ndcg_2.append(_ndcg_at_k(titles, gold_titles, 2))
            ndcg_5.append(_ndcg_at_k(titles, gold_titles, 5))
            continue

        if mode == "text_only":
            row_recall = _text_recall(contexts, gold_titles)
            recall_2.append(row_recall)
            recall_5.append(row_recall)
            recall_text.append(row_recall)
            continue

    structured_count = mode_counts["structured_ranked"]
    text_count = mode_counts["text_only"]
    if structured_count > 0 and text_count == 0:
        recall_mode = "structured_ranked"
    elif text_count > 0 and structured_count == 0:
        recall_mode = "text"
    elif text_count > 0 and structured_count > 0:
        recall_mode = "mixed"
    else:
        recall_mode = "invalid"

    result: Dict[str, Any] = {
        "recall@2": _mean(recall_2),
        "recall@5": _mean(recall_5),
        "ie@2": _mean(ie_2),
        "ie@5": _mean(ie_5),
        "ndcg@2": _mean(ndcg_2),
        "ndcg@5": _mean(ndcg_5),
        "recall_text": _mean(recall_text),
        "recall_mode": recall_mode,
        "id_overlap_count": id_overlap_count,
        "evaluated_count": evaluated_count,
        "expected_count": int(expected_count),
        "alignment_ok": alignment_ok,
        "mode_counts": mode_counts,
        "gold_count": len(gold_map),
        "pred_count": len(pred_map),
        "duplicate_pred_count": duplicate_pred_count,
        "count": evaluated_count,
    }

    if not alignment_ok:
        for key in ("recall@2", "recall@5", "ie@2", "ie@5", "ndcg@2", "ndcg@5", "recall_text"):
            result[key] = None

    if recall_mode in {"text", "mixed", "invalid"}:
        result["ie@2"] = None
        result["ie@5"] = None
        result["ndcg@2"] = None
        result["ndcg@5"] = None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation (v3) with schema/alignment awareness.")
    parser.add_argument("--pred_file", required=True, type=Path)
    parser.add_argument("--gold_file", required=True, type=Path)
    parser.add_argument("--expected_count", type=int, default=500)
    args = parser.parse_args()

    result = evaluate_retrieval(
        pred_file=args.pred_file,
        gold_file=args.gold_file,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
