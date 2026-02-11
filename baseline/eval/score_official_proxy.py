#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import iter_jsonl, load_predictions  # noqa: E402
from score_squad_style import score_rows  # noqa: E402


def score_musique_proxy(pred_path: Path, qa_path: Path) -> Dict:
    qa_rows = list(iter_jsonl(qa_path))
    pred_map = load_predictions(pred_path)

    answerable_count = sum(1 for row in qa_rows if bool(row.get("answerable", True)))
    primary = score_rows(
        qa_rows,
        pred_map,
        filter_answerable=True,
        use_aliases=False,
        restrict_to_pred_ids=True,
    )
    alt = score_rows(
        qa_rows,
        pred_map,
        filter_answerable=True,
        use_aliases=True,
        restrict_to_pred_ids=True,
    )
    return {
        "dataset": "musique",
        "count_total": len(qa_rows),
        "count_answerable": answerable_count,
        "count_unanswerable": len(qa_rows) - answerable_count,
        "primary": primary,
        "alt": alt,
    }


def score_2wiki_proxy(pred_path: Path, qa_path: Path) -> Dict:
    qa_rows = list(iter_jsonl(qa_path))
    pred_map = load_predictions(pred_path)
    proxy = score_rows(
        qa_rows,
        pred_map,
        filter_answerable=False,
        use_aliases=True,
        restrict_to_pred_ids=True,
    )
    return {
        "dataset": "2wiki",
        "count_total": len(qa_rows),
        "proxy": proxy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score official/proxy comparable metrics")
    parser.add_argument("--pred", required=True, help="Path to pred.jsonl")
    parser.add_argument("--qa", required=True, help="Path to qa.jsonl")
    parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    qa_path = Path(args.qa)

    if args.dataset == "musique":
        result = score_musique_proxy(pred_path, qa_path)
    else:
        result = score_2wiki_proxy(pred_path, qa_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
