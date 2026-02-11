#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def normalize_answer(text: str) -> str:
    import re
    import string

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text or "").lower())))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)

    if pred in {"yes", "no", "noanswer"} and pred != gold:
        return zero
    if gold in {"yes", "no", "noanswer"} and pred != gold:
        return zero

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1), float(precision), float(recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics: Dict[str, float], prediction: str, gold: str) -> Tuple[float, float, float]:
    em = exact_match_score(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    metrics["em"] += float(em)
    metrics["f1"] += float(f1)
    metrics["prec"] += float(precision)
    metrics["recall"] += float(recall)
    return em, precision, recall


def update_sp(metrics: Dict[str, float], prediction: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]) -> Tuple[float, float, float]:
    pred_set = set(map(tuple, prediction))
    gold_set = set(map(tuple, gold))

    tp = sum(1 for item in pred_set if item in gold_set)
    fp = sum(1 for item in pred_set if item not in gold_set)
    fn = sum(1 for item in gold_set if item not in pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += precision
    metrics["sp_recall"] += recall
    return em, precision, recall


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported gold format: {path}")


def load_pred_map(pred_path: Path) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    for row in iter_jsonl(pred_path):
        qid = str(row.get("id") or row.get("_id") or "").strip()
        if not qid:
            continue
        pred_map[qid] = str(row.get("pred") or "").strip()
    return pred_map


def build_hotpot_payload(pred_map: Dict[str, str], gold_rows: Sequence[Dict]) -> Dict[str, Dict]:
    answer = {}
    sp = {}
    for row in gold_rows:
        qid = str(row.get("_id") or row.get("id") or "").strip()
        if not qid:
            continue
        if qid in pred_map:
            answer[qid] = pred_map.get(qid, "")
            sp[qid] = []
    return {"answer": answer, "sp": sp}


def evaluate_hotpot_payload(
    prediction: Dict[str, Dict],
    gold_rows: Sequence[Dict],
    *,
    restrict_to_pred_ids: bool,
) -> Dict[str, float]:
    metrics = {
        "em": 0.0,
        "f1": 0.0,
        "prec": 0.0,
        "recall": 0.0,
        "sp_em": 0.0,
        "sp_f1": 0.0,
        "sp_prec": 0.0,
        "sp_recall": 0.0,
        "joint_em": 0.0,
        "joint_f1": 0.0,
        "joint_prec": 0.0,
        "joint_recall": 0.0,
    }

    answer_map = prediction.get("answer", {})
    sp_map = prediction.get("sp", {})

    if restrict_to_pred_ids:
        eval_rows = []
        for row in gold_rows:
            qid = str(row.get("_id") or row.get("id") or "").strip()
            if qid and qid in answer_map:
                eval_rows.append(row)
    else:
        eval_rows = list(gold_rows)

    for row in eval_rows:
        qid = str(row.get("_id") or row.get("id") or "").strip()
        if not qid:
            continue

        can_eval_joint = True
        if qid not in answer_map:
            can_eval_joint = False
            em, precision, recall = 0.0, 0.0, 0.0
        else:
            em, precision, recall = update_answer(
                metrics,
                str(answer_map.get(qid) or ""),
                str(row.get("answer") or ""),
            )

        if qid not in sp_map:
            can_eval_joint = False
            sp_em, sp_precision, sp_recall = 0.0, 0.0, 0.0
        else:
            sp_em, sp_precision, sp_recall = update_sp(
                metrics,
                sp_map.get(qid) or [],
                row.get("supporting_facts") or [],
            )

        if can_eval_joint:
            joint_precision = precision * sp_precision
            joint_recall = recall * sp_recall
            if joint_precision + joint_recall > 0:
                joint_f1 = (2 * joint_precision * joint_recall) / (joint_precision + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em
            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_precision
            metrics["joint_recall"] += joint_recall

    count = float(len(eval_rows))
    if count <= 0:
        return {**metrics, "count": 0}

    for key in list(metrics.keys()):
        metrics[key] /= count
    metrics["count"] = int(count)
    return metrics


def score_hotpot(
    pred_path: Path,
    gold_path: Path,
    *,
    restrict_to_pred_ids: bool = True,
) -> Dict[str, float]:
    gold_rows = load_gold(gold_path)
    pred_map = load_pred_map(pred_path)
    payload = build_hotpot_payload(pred_map, gold_rows)
    return evaluate_hotpot_payload(
        payload,
        gold_rows,
        restrict_to_pred_ids=restrict_to_pred_ids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Hotpot predictions with official-style metrics")
    parser.add_argument("--pred", required=True, help="Path to pred.jsonl")
    parser.add_argument(
        "--gold",
        default="data/hotpot_dev_distractor_500_jsonl_official_gold.json",
        help="Gold file (.json or .jsonl)",
    )
    parser.add_argument("--no_restrict_to_pred_ids", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    gold_path = Path(args.gold)
    result = score_hotpot(
        pred_path,
        gold_path,
        restrict_to_pred_ids=not bool(args.no_restrict_to_pred_ids),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
