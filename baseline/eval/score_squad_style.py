#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import best_score_against_golds, iter_jsonl, load_predictions  # noqa: E402


def _unique_non_empty(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def score_rows(
    qa_rows: List[Dict],
    pred_map: Dict[str, str],
    *,
    filter_answerable: bool,
    use_aliases: bool,
    restrict_to_pred_ids: bool,
) -> Dict[str, float]:
    total = 0
    sum_em = 0.0
    sum_f1 = 0.0

    for row in qa_rows:
        if filter_answerable and not bool(row.get("answerable", True)):
            continue

        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        if restrict_to_pred_ids and qid not in pred_map:
            continue

        answer = str(row.get("answer") or "").strip()
        aliases = row.get("answer_aliases") or []
        if not isinstance(aliases, list):
            aliases = []

        golds = [answer]
        if use_aliases:
            golds.extend(str(item) for item in aliases)
        golds = _unique_non_empty(golds)
        if not golds:
            golds = [""]

        pred = pred_map.get(qid, "")
        best = best_score_against_golds(pred, golds)

        total += 1
        sum_em += best["em"]
        sum_f1 += best["f1"]

    if total == 0:
        return {"em": 0.0, "f1": 0.0, "count": 0}
    return {"em": sum_em / total, "f1": sum_f1 / total, "count": total}


def score_squad_style(
    pred_path: Path,
    qa_path: Path,
    *,
    filter_answerable: bool = False,
    use_aliases: bool = True,
    restrict_to_pred_ids: bool = True,
) -> Dict[str, float]:
    qa_rows = list(iter_jsonl(qa_path))
    pred_map = load_predictions(pred_path)
    return score_rows(
        qa_rows,
        pred_map,
        filter_answerable=filter_answerable,
        use_aliases=use_aliases,
        restrict_to_pred_ids=restrict_to_pred_ids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score predictions with SQuAD-style EM/F1")
    parser.add_argument("--pred", required=True, help="Path to pred.jsonl")
    parser.add_argument("--qa", required=True, help="Path to qa.jsonl")
    parser.add_argument("--filter_answerable", action="store_true")
    parser.add_argument("--no_aliases", action="store_true")
    parser.add_argument("--no_restrict_to_pred_ids", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_squad_style(
        Path(args.pred),
        Path(args.qa),
        filter_answerable=bool(args.filter_answerable),
        use_aliases=not bool(args.no_aliases),
        restrict_to_pred_ids=not bool(args.no_restrict_to_pred_ids),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
