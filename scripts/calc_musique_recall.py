import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_K_LIST = [1, 2, 3, 5, 10]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def _get_context_list(row: Dict[str, Any], context_field: str) -> List[Dict[str, Any]]:
    preferred = row.get(context_field)
    if isinstance(preferred, list):
        return [item for item in preferred if isinstance(item, dict)]

    if context_field == "retrieved_context_topk":
        fallback = row.get("retrieved_context")
        if isinstance(fallback, list):
            return [item for item in fallback if isinstance(item, dict)]
    return []


def _extract_titles(context_rows: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    for item in context_rows:
        title = item.get("title") or item.get("doc_title")
        title_text = str(title or "").strip()
        if title_text:
            titles.append(title_text)
    return titles


def calculate_recall(
    file_path: str,
    *,
    context_field: str = "retrieved_context_raw",
    k_list: List[int] = None,
) -> Tuple[Dict[str, float], int]:
    ks = list(k_list or DEFAULT_K_LIST)
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    count_with_sp = 0

    try:
        for row in _iter_jsonl(file_path):
            gold_sp = row.get("gold_sp") or []
            gold_titles = {
                str(item[0]).strip()
                for item in gold_sp
                if isinstance(item, list) and len(item) >= 1 and str(item[0]).strip()
            }
            if not gold_titles:
                continue
            count_with_sp += 1

            context_rows = _get_context_list(row, context_field)
            retrieved_titles = _extract_titles(context_rows)

            for k in ks:
                top_k_titles = set(retrieved_titles[:k])
                hit = sum(1 for title in gold_titles if title in top_k_titles)
                recalls[k].append(hit / len(gold_titles))
    except FileNotFoundError:
        return {}, 0

    results = {f"R@{k}": _mean(recalls[k]) for k in ks}
    return results, count_with_sp


def _basename(path: str) -> str:
    return os.path.basename(path).replace("pred_dev_", "").replace(".jsonl", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute retrieval title recall@k from prediction jsonl.")
    parser.add_argument("files", nargs="+", help="Prediction JSONL files")
    parser.add_argument(
        "--context_field",
        default="retrieved_context_raw",
        help="Context field to evaluate (e.g. retrieved_context_raw or retrieved_context_topk)",
    )
    args = parser.parse_args()

    print("| File | Context | R@1 | R@2 | R@3 | R@5 | R@10 | Count (with SP) |")
    print("|---|---|---|---|---|---|---|---|")
    for file_path in args.files:
        metrics, count = calculate_recall(file_path, context_field=args.context_field)
        if not metrics:
            print(f"| {_basename(file_path)} | {args.context_field} | N/A | N/A | N/A | N/A | N/A | 0 |")
            continue
        print(
            f"| {_basename(file_path)} | {args.context_field} | "
            f"{metrics.get('R@1', 0.0):.4f} | {metrics.get('R@2', 0.0):.4f} | "
            f"{metrics.get('R@3', 0.0):.4f} | {metrics.get('R@5', 0.0):.4f} | "
            f"{metrics.get('R@10', 0.0):.4f} | {count} |"
        )


if __name__ == "__main__":
    main()
