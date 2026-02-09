import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

Unit = Union[str, Tuple[str, int]]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _clean_title(value: Any) -> str:
    return str(value or "").strip()


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid index")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("non-integer float")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty index")
    return int(text)


def _extract_gold_units(row: Dict[str, Any], unit: str) -> List[Unit]:
    gold_sp = row.get("gold_sp") or []
    units: List[Unit] = []
    for item in gold_sp:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        title = _clean_title(item[0])
        if not title:
            continue
        if unit == "title":
            units.append(title)
            continue
        if len(item) < 2:
            continue
        try:
            paragraph_idx = _to_int(item[1])
        except Exception:
            continue
        units.append((title, paragraph_idx))
    return units


def _extract_retrieved_units(row: Dict[str, Any], context_field: str, unit: str) -> List[Unit]:
    contexts = row.get(context_field)
    if not isinstance(contexts, list):
        return []

    units: List[Unit] = []
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        title = _clean_title(ctx.get("title") or ctx.get("doc_title"))
        if not title:
            continue
        if unit == "title":
            units.append(title)
            continue

        idx_value = ctx.get("paragraph_idx")
        try:
            paragraph_idx = _to_int(idx_value)
        except Exception:
            continue
        units.append((title, paragraph_idx))
    return units


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def calculate_recall_strict(
    file_path: str,
    *,
    k_list: Sequence[int],
    context_field: str = "retrieved_context_topk",
    unit: str = "pair",
) -> Dict[str, Any]:
    macro_scores: Dict[int, List[float]] = {k: [] for k in k_list}
    full_hit_flags: Dict[int, List[float]] = {k: [] for k in k_list}
    micro_hits: Dict[int, int] = {k: 0 for k in k_list}
    total_gold_units = 0
    total_rows = 0
    valid_rows = 0
    skipped_no_gold = 0

    for row in _iter_jsonl(file_path):
        total_rows += 1
        gold_units = set(_extract_gold_units(row, unit))
        if not gold_units:
            skipped_no_gold += 1
            continue

        valid_rows += 1
        total_gold_units += len(gold_units)
        retrieved_units = _extract_retrieved_units(row, context_field, unit)
        for k in k_list:
            topk = set(retrieved_units[:k])
            hit = len(gold_units & topk)
            recall = hit / len(gold_units)
            macro_scores[k].append(recall)
            full_hit_flags[k].append(1.0 if hit == len(gold_units) else 0.0)
            micro_hits[k] += hit

    results: Dict[str, Any] = {
        "file": file_path,
        "unit": unit,
        "context_field": context_field,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "skipped_no_gold": skipped_no_gold,
        "total_gold_units": total_gold_units,
        "metrics": {},
    }

    for k in k_list:
        macro = _mean(macro_scores[k])
        micro = (micro_hits[k] / total_gold_units) if total_gold_units else 0.0
        full_hit = _mean(full_hit_flags[k])
        results["metrics"][f"R@{k}_macro"] = macro
        results["metrics"][f"R@{k}_micro"] = micro
        results["metrics"][f"Full@{k}"] = full_hit

    return results


def _basename(path: str) -> str:
    return os.path.basename(path).replace("pred_dev_", "").replace(".jsonl", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict MuSiQue recall@k calculator on title+paragraph_idx (or title only)."
    )
    parser.add_argument("files", nargs="+", help="Prediction JSONL files")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[2, 5],
        help="Recall cutoffs, e.g. --k 2 5",
    )
    parser.add_argument(
        "--context_field",
        default="retrieved_context_topk",
        help="Context field to evaluate, default: retrieved_context_topk",
    )
    parser.add_argument(
        "--unit",
        choices=["pair", "title"],
        default="pair",
        help="pair = (title, paragraph_idx), title = title only",
    )
    args = parser.parse_args()

    ks = sorted({k for k in args.k if k > 0})
    if not ks:
        raise ValueError("No valid k values.")

    metric_headers: List[str] = []
    for k in ks:
        metric_headers.append(f"R@{k}_macro")
    for k in ks:
        metric_headers.append(f"R@{k}_micro")
    for k in ks:
        metric_headers.append(f"Full@{k}")

    print("| File | Unit | Context | Valid Rows | Skipped(no gold_sp) | " + " | ".join(metric_headers) + " |")
    print("|---|---|---|---|---|" + "---|" * len(metric_headers))

    for file_path in args.files:
        stats = calculate_recall_strict(
            file_path,
            k_list=ks,
            context_field=args.context_field,
            unit=args.unit,
        )
        metrics = stats["metrics"]
        values: List[str] = []
        for key in metric_headers:
            values.append(f"{metrics.get(key, 0.0):.6f}")
        print(
            f"| {_basename(file_path)} | {args.unit} | {args.context_field} | "
            f"{stats['valid_rows']} | {stats['skipped_no_gold']} | {' | '.join(values)} |"
        )


if __name__ == "__main__":
    main()
