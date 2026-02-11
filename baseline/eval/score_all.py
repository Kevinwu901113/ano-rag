#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import empty_rate, load_predictions, refusal_rate  # noqa: E402
from score_hotpot import score_hotpot  # noqa: E402
from score_official_proxy import score_2wiki_proxy, score_musique_proxy  # noqa: E402
from score_squad_style import score_squad_style  # noqa: E402


CSV_COLUMNS = [
    "method",
    "dataset",
    "llm_backend",
    "metric_family",
    "em",
    "f1",
    "count",
    "empty_rate",
    "refusal_rate",
    "ts",
]


def discover_runs(results_root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    for pred_path in sorted(results_root.glob("*/*/*/pred.jsonl")):
        rel = pred_path.relative_to(results_root)
        if len(rel.parts) != 4:
            continue
        method, dataset, backend, filename = rel.parts
        if filename != "pred.jsonl":
            continue
        yield method, dataset, backend, pred_path


def build_row(
    *,
    method: str,
    dataset: str,
    llm_backend: str,
    metric_family: str,
    em: float,
    f1: float,
    count: int,
    empty: float,
    refusal: float,
    ts: str,
) -> Dict[str, object]:
    return {
        "method": method,
        "dataset": dataset,
        "llm_backend": llm_backend,
        "metric_family": metric_family,
        "em": f"{float(em):.6f}",
        "f1": f"{float(f1):.6f}",
        "count": int(count),
        "empty_rate": f"{float(empty):.6f}",
        "refusal_rate": f"{float(refusal):.6f}",
        "ts": ts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline metrics to CSV")
    parser.add_argument("--results_root", default="baseline/results")
    parser.add_argument("--data_root", default="baseline/data")
    parser.add_argument(
        "--hotpot_gold",
        default="data/hotpot_dev_distractor_500_jsonl_official_gold.json",
    )
    parser.add_argument("--out_csv", default="baseline/results/metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    hotpot_gold = Path(args.hotpot_gold)
    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    rows: List[Dict[str, object]] = []
    for method, dataset, backend, pred_path in discover_runs(results_root):
        qa_path = data_root / dataset / "qa.jsonl"
        if not qa_path.exists():
            print(f"[warn] missing qa file for {dataset}: {qa_path}")
            continue

        pred_map = load_predictions(pred_path)
        empty = empty_rate(pred_map)
        refusal = refusal_rate(pred_map)

        squad = score_squad_style(pred_path, qa_path, filter_answerable=False, use_aliases=True)
        rows.append(
            build_row(
                method=method,
                dataset=dataset,
                llm_backend=backend,
                metric_family="squad_style",
                em=squad["em"],
                f1=squad["f1"],
                count=squad["count"],
                empty=empty,
                refusal=refusal,
                ts=ts,
            )
        )

        if dataset == "hotpotqa":
            hotpot = score_hotpot(pred_path, hotpot_gold)
            rows.append(
                build_row(
                    method=method,
                    dataset=dataset,
                    llm_backend=backend,
                    metric_family="hotpot_official",
                    em=hotpot["em"],
                    f1=hotpot["f1"],
                    count=int(hotpot["count"]),
                    empty=empty,
                    refusal=refusal,
                    ts=ts,
                )
            )
        elif dataset == "musique":
            proxy = score_musique_proxy(pred_path, qa_path)
            rows.append(
                build_row(
                    method=method,
                    dataset=dataset,
                    llm_backend=backend,
                    metric_family="official_proxy_primary",
                    em=proxy["primary"]["em"],
                    f1=proxy["primary"]["f1"],
                    count=proxy["primary"]["count"],
                    empty=empty,
                    refusal=refusal,
                    ts=ts,
                )
            )
            rows.append(
                build_row(
                    method=method,
                    dataset=dataset,
                    llm_backend=backend,
                    metric_family="official_proxy_alt",
                    em=proxy["alt"]["em"],
                    f1=proxy["alt"]["f1"],
                    count=proxy["alt"]["count"],
                    empty=empty,
                    refusal=refusal,
                    ts=ts,
                )
            )
        elif dataset == "2wiki":
            proxy = score_2wiki_proxy(pred_path, qa_path)
            rows.append(
                build_row(
                    method=method,
                    dataset=dataset,
                    llm_backend=backend,
                    metric_family="official_proxy",
                    em=proxy["proxy"]["em"],
                    f1=proxy["proxy"]["f1"],
                    count=proxy["proxy"]["count"],
                    empty=empty,
                    refusal=refusal,
                    ts=ts,
                )
            )

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({"rows": len(rows), "out_csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
