from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REFUSAL_KEYWORDS = (
    "insufficient evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "i don't know",
    "unknown",
)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(text or "")))))


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


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_score_against_golds(prediction: str, golds: Sequence[str]) -> Dict[str, float]:
    if not golds:
        return {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}

    best = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    for gold in golds:
        em = exact_match(prediction, gold)
        f1, prec, recall = f1_score(prediction, gold)
        if f1 > best["f1"] or (f1 == best["f1"] and em > best["em"]):
            best = {"em": em, "f1": f1, "prec": prec, "recall": recall}
    return best


def load_predictions(path: Path) -> Dict[str, str]:
    pred_map: Dict[str, str] = {}
    for row in iter_jsonl(path):
        qid = str(row.get("id") or row.get("_id") or "").strip()
        if not qid:
            continue
        pred_map[qid] = str(row.get("pred", "")).strip()
    return pred_map


def refusal_rate(preds: Dict[str, str]) -> float:
    if not preds:
        return 0.0
    refused = 0
    for value in preds.values():
        norm = str(value or "").strip().lower()
        if any(key in norm for key in REFUSAL_KEYWORDS):
            refused += 1
    return refused / len(preds)


def empty_rate(preds: Dict[str, str]) -> float:
    if not preds:
        return 0.0
    empty = sum(1 for value in preds.values() if not str(value or "").strip())
    return empty / len(preds)
