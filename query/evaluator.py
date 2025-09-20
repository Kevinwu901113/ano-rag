import json
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _normalize_answer(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _support_f1(predicted: Sequence[int], gold: Sequence[int]) -> float:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    intersection = len(pred_set & gold_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _coverage(predicted: Sequence[int], gold: Sequence[int]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    return len(gold_set & set(predicted)) / len(gold_set)


def _candidate_idx(candidate_id: Any) -> Optional[int]:
    if candidate_id is None:
        return None
    if isinstance(candidate_id, int):
        return candidate_id
    text = str(candidate_id)
    if "__" in text:
        text = text.split("__")[-1]
    try:
        return int(text)
    except ValueError:
        return None


def _ensure_int_list(values: Optional[Iterable[Any]]) -> List[int]:
    if not values:
        return []
    result: List[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _coerce_support_indices(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            try:
                return [int(raw)]
            except (TypeError, ValueError):
                return []
        else:
            return _coerce_support_indices(parsed)
    if isinstance(raw, dict):
        candidates: List[int] = []
        for key in ("idx", "index", "paragraph_idx", "paragraph_id", "support_idx"):
            if key in raw:
                try:
                    candidates.append(int(raw[key]))
                except (TypeError, ValueError):
                    continue
        if candidates:
            return candidates
        for key in ("indices", "idxs", "values"):
            if key in raw:
                return _coerce_support_indices(raw[key])
        return []
    if isinstance(raw, (list, tuple)):
        collected: List[int] = []
        for item in raw:
            collected.extend(_coerce_support_indices(item))
        return collected
    try:
        return [int(raw)]
    except (TypeError, ValueError):
        return []


def _extract_gold_answers(reference: Optional[Dict[str, Any]]) -> List[str]:
    if not reference:
        return []
    answer_keys = ["answers", "answer", "gold_answers", "gold_answer", "reference_answers"]
    for key in answer_keys:
        if key not in reference:
            continue
        raw_value = reference[key]
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            return [raw_value]
        if isinstance(raw_value, (list, tuple)):
            return [str(item) for item in raw_value if item]
        return [str(raw_value)]
    return []


def _extract_gold_support_idxs(reference: Optional[Dict[str, Any]]) -> List[int]:
    if not reference:
        return []
    support_keys = [
        "support_idxs",
        "support_idx",
        "supporting_facts",
        "supporting_indices",
        "supporting_passages",
        "gold_support_idxs",
        "gold_support_idx",
    ]
    for key in support_keys:
        if key not in reference:
            continue
        idxs = _coerce_support_indices(reference[key])
        if idxs:
            return _dedupe_preserve_order(idxs)
    return []


class RetrievalEvaluator:
    """Accumulates retrieval and answer metrics across queries."""

    def __init__(self, topk_eval: int = 50, results_dir: str = "results") -> None:
        self.topk_eval = topk_eval
        self.results_dir = Path(results_dir)
        self.records: List[Dict[str, Any]] = []

    def add_record(
        self,
        *,
        query_id: str,
        question: str,
        retrieval_candidates: Sequence[Any],
        predicted_answer: Optional[str],
        predicted_support_idxs: Optional[Sequence[Any]],
        reference: Optional[Dict[str, Any]] = None,
    ) -> None:
        prepared_candidates = self._prepare_candidates(retrieval_candidates)
        record = {
            "query_id": query_id,
            "question": question,
            "retrieval_candidates": prepared_candidates,
            "predicted_answer": (predicted_answer or "").strip(),
            "predicted_support_idxs": _ensure_int_list(predicted_support_idxs),
            "gold_answers": _extract_gold_answers(reference),
            "gold_support_idxs": _extract_gold_support_idxs(reference),
        }
        self.records.append(record)

    def _prepare_candidates(self, candidates: Sequence[Any]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        if not candidates:
            return prepared
        for rank, candidate in enumerate(candidates, start=1):
            if rank > self.topk_eval:
                break
            cand_id: Any = None
            score: Optional[float] = None
            if isinstance(candidate, dict):
                cand_id = candidate.get("id") or candidate.get("candidate_id") or candidate.get("doc_id")
                raw_score = candidate.get("score") or candidate.get("weight")
            elif isinstance(candidate, (list, tuple)) and candidate:
                cand_id = candidate[0]
                raw_score = candidate[1] if len(candidate) > 1 else None
            else:
                cand_id = candidate
                raw_score = None
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
            prepared.append({"id": cand_id, "score": score})
        return prepared

    def _per_query_metrics(self, record: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        gold_supports = record.get("gold_support_idxs", [])
        pred_supports = record.get("predicted_support_idxs", [])
        candidate_indices = [_candidate_idx(item.get("id")) for item in record.get("retrieval_candidates", [])]
        candidate_indices = [idx for idx in candidate_indices if idx is not None]
        recall_key = f"recall_at_{self.topk_eval}"
        mrr_key = f"mrr_at_{self.topk_eval}"
        if gold_supports:
            if candidate_indices:
                metrics[recall_key] = _coverage(candidate_indices, gold_supports)
                metrics[mrr_key] = self._mrr(candidate_indices, gold_supports)
            else:
                metrics[recall_key] = 0.0
                metrics[mrr_key] = 0.0
            metrics["support_index_coverage"] = _coverage(pred_supports, gold_supports)
            metrics["support_f1"] = _support_f1(pred_supports, gold_supports)
        gold_answers = record.get("gold_answers", [])
        if gold_answers:
            pred_answer = record.get("predicted_answer", "")
            metrics["answer_f1"] = max(_answer_f1(pred_answer, gold) for gold in gold_answers)
        return metrics

    @staticmethod
    def _mrr(candidates: Sequence[int], gold: Sequence[int]) -> float:
        gold_set = set(gold)
        for rank, idx in enumerate(candidates, start=1):
            if idx in gold_set:
                return 1.0 / rank
        return 0.0

    def compute_metrics(self) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        per_query: List[Dict[str, Any]] = []
        aggregates: Dict[str, List[float]] = {}
        for record in self.records:
            metrics = self._per_query_metrics(record)
            question = record.get("question") or ""
            preview = question if len(question) <= 160 else f"{question[:157]}..."
            per_query.append({
                "id": record.get("query_id"),
                "question_preview": preview,
                "metrics": metrics,
            })
            for key, value in metrics.items():
                aggregates.setdefault(key, []).append(value)
        summary: Dict[str, float] = {}
        metric_name_map = {
            f"recall_at_{self.topk_eval}": f"R@{self.topk_eval}",
            f"mrr_at_{self.topk_eval}": f"MRR@{self.topk_eval}",
            "answer_f1": "AnswerF1",
            "support_f1": "SupportF1",
            "support_index_coverage": "SupportIndexCoverage",
        }
        for key, values in aggregates.items():
            if not values:
                continue
            mapped_key = metric_name_map.get(key, key)
            summary[mapped_key] = sum(values) / len(values)
        return summary, per_query

    def finalize(self) -> Tuple[Optional[Path], Dict[str, Any]]:
        if not self.records:
            return None, {}
        summary, per_query = self.compute_metrics()
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "topk_eval": self.topk_eval,
            "num_queries": len(self.records),
            "metrics": summary,
            "per_query": per_query,
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.results_dir / f"baseline_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return output_path, report
