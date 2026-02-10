#!/usr/bin/env python3
import argparse
import collections
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(value: str) -> str:
        return " ".join([t for t in value.split() if t not in ["a", "an", "the"]])

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def calculate_ndcg(retrieved_items: List[Tuple[str, int]], gold_set: Set[Tuple[str, int]], k: int) -> float:
    relevance = []
    seen = set()
    for idx in range(min(k, len(retrieved_items))):
        item = tuple(retrieved_items[idx])
        if item in gold_set and item not in seen:
            relevance.append(1)
            seen.add(item)
        else:
            relevance.append(0)

    dcg = 0.0
    for idx, rel in enumerate(relevance):
        dcg += rel / np.log2(idx + 2)

    ideal_k = min(len(gold_set), k)
    idcg = 0.0
    for idx in range(ideal_k):
        idcg += 1.0 / np.log2(idx + 2)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def calculate_recall(retrieved_items: List[Tuple[str, int]], gold_set: Set[Tuple[str, int]], k: int) -> float:
    if not gold_set:
        return 0.0
    retrieved_k = [tuple(x) for x in retrieved_items[:k]]
    hits = len(set(retrieved_k) & gold_set)
    return hits / len(gold_set)


def _normalize_fact_list(raw: Any) -> List[Tuple[str, int]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0] or "").strip()
        if not title:
            continue
        try:
            sent_idx = int(item[1])
        except (TypeError, ValueError):
            continue
        key = (title, sent_idx)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _facts_from_contexts(contexts: Any) -> List[Tuple[str, int]]:
    if not isinstance(contexts, list):
        return []
    facts: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()
    for row in contexts:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        sentence_idx = row.get("sentence_idx")
        try:
            sentence_idx = int(sentence_idx)
        except (TypeError, ValueError):
            continue
        key = (title, sentence_idx)
        if key in seen:
            continue
        seen.add(key)
        facts.append(key)
    return facts


def _extract_stage_contexts(pred: Dict[str, Any], stage_name: str) -> List[Dict[str, Any]]:
    stages = pred.get("retrieval_stages")
    if isinstance(stages, dict):
        stage = stages.get(stage_name)
        if isinstance(stage, dict):
            contexts = stage.get("contexts_topk")
            if isinstance(contexts, list):
                return contexts
    if stage_name == "final_with_fallback":
        fallback_contexts = pred.get("retrieved_context_topk")
        if isinstance(fallback_contexts, list):
            return fallback_contexts
    return []


def _stage_available(pred: Dict[str, Any], stage_name: str) -> bool:
    stages = pred.get("retrieval_stages")
    if isinstance(stages, dict):
        stage = stages.get(stage_name)
        if isinstance(stage, dict) and "available" in stage:
            return bool(stage.get("available"))
    return bool(_extract_stage_contexts(pred, stage_name))


def _extract_final_facts(pred: Dict[str, Any]) -> List[Tuple[str, int]]:
    facts = _facts_from_contexts(_extract_stage_contexts(pred, "final_with_fallback"))
    if facts:
        return facts
    legacy = pred.get("pred_sp_topk")
    if not legacy:
        legacy = pred.get("pred_sp")
    return _normalize_fact_list(legacy)


def _extract_stage_facts(pred: Dict[str, Any], stage_name: str) -> List[Tuple[str, int]]:
    return _facts_from_contexts(_extract_stage_contexts(pred, stage_name))


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def evaluate_run(pred_file: Path, gold_data: Dict[str, Any]) -> Dict[str, float]:
    with pred_file.open("r", encoding="utf-8") as handle:
        preds = [json.loads(line) for line in handle if line.strip()]

    metrics: Dict[str, List[float]] = {
        "em": [],
        "f1": [],
        "final_recall@2": [],
        "final_recall@5": [],
        "final_recall@10": [],
        "final_ndcg@10": [],
        "stage1_recall@2": [],
        "stage1_recall@5": [],
        "stage1_recall@10": [],
        "stage1_ndcg@10": [],
        "stage2_recall@2": [],
        "stage2_recall@5": [],
        "stage2_recall@10": [],
        "stage2_ndcg@10": [],
        "delta_final_minus_stage2@2": [],
        "delta_final_minus_stage2@5": [],
        "delta_final_minus_stage2@10": [],
    }
    stats = {
        "count": 0,
        "stage1_available_count": 0,
        "stage2_available_count": 0,
        "fallback_used_count": 0,
    }

    for pred in preds:
        qid = pred.get("_id")
        if qid not in gold_data:
            continue
        stats["count"] += 1

        gold_entry = gold_data[qid]
        gold_answer = str(gold_entry.get("answer") or "")
        gold_sp = set(_normalize_fact_list(gold_entry.get("supporting_facts")))

        pred_answer = str(pred.get("short_answer", pred.get("answer", "")) or "")
        em = exact_match_score(pred_answer, gold_answer)
        f1, _, _ = f1_score(pred_answer, gold_answer)
        metrics["em"].append(em)
        metrics["f1"].append(f1)

        final_facts = _extract_final_facts(pred)
        final_r2 = calculate_recall(final_facts, gold_sp, 2)
        final_r5 = calculate_recall(final_facts, gold_sp, 5)
        final_r10 = calculate_recall(final_facts, gold_sp, 10)
        metrics["final_recall@2"].append(final_r2)
        metrics["final_recall@5"].append(final_r5)
        metrics["final_recall@10"].append(final_r10)
        metrics["final_ndcg@10"].append(calculate_ndcg(final_facts, gold_sp, 10))

        stage1_available = _stage_available(pred, "stage1")
        if stage1_available:
            stats["stage1_available_count"] += 1
            stage1_facts = _extract_stage_facts(pred, "stage1")
            metrics["stage1_recall@2"].append(calculate_recall(stage1_facts, gold_sp, 2))
            metrics["stage1_recall@5"].append(calculate_recall(stage1_facts, gold_sp, 5))
            metrics["stage1_recall@10"].append(calculate_recall(stage1_facts, gold_sp, 10))
            metrics["stage1_ndcg@10"].append(calculate_ndcg(stage1_facts, gold_sp, 10))

        stage2_available = _stage_available(pred, "stage2_no_fallback")
        if stage2_available:
            stats["stage2_available_count"] += 1
            stage2_facts = _extract_stage_facts(pred, "stage2_no_fallback")
            stage2_r2 = calculate_recall(stage2_facts, gold_sp, 2)
            stage2_r5 = calculate_recall(stage2_facts, gold_sp, 5)
            stage2_r10 = calculate_recall(stage2_facts, gold_sp, 10)
            metrics["stage2_recall@2"].append(stage2_r2)
            metrics["stage2_recall@5"].append(stage2_r5)
            metrics["stage2_recall@10"].append(stage2_r10)
            metrics["stage2_ndcg@10"].append(calculate_ndcg(stage2_facts, gold_sp, 10))
            metrics["delta_final_minus_stage2@2"].append(final_r2 - stage2_r2)
            metrics["delta_final_minus_stage2@5"].append(final_r5 - stage2_r5)
            metrics["delta_final_minus_stage2@10"].append(final_r10 - stage2_r10)

        fallback_used = bool(((pred.get("intermediate") or {}).get("fallback") or {}).get("used", False))
        if fallback_used:
            stats["fallback_used_count"] += 1

    total = int(stats["count"])
    aggregated = {
        "em": _mean(metrics["em"]),
        "f1": _mean(metrics["f1"]),
        "final_recall@2": _mean(metrics["final_recall@2"]),
        "final_recall@5": _mean(metrics["final_recall@5"]),
        "final_recall@10": _mean(metrics["final_recall@10"]),
        "final_ndcg@10": _mean(metrics["final_ndcg@10"]),
        "stage1_recall@2": _mean(metrics["stage1_recall@2"]),
        "stage1_recall@5": _mean(metrics["stage1_recall@5"]),
        "stage1_recall@10": _mean(metrics["stage1_recall@10"]),
        "stage1_ndcg@10": _mean(metrics["stage1_ndcg@10"]),
        "stage2_recall@2": _mean(metrics["stage2_recall@2"]),
        "stage2_recall@5": _mean(metrics["stage2_recall@5"]),
        "stage2_recall@10": _mean(metrics["stage2_recall@10"]),
        "stage2_ndcg@10": _mean(metrics["stage2_ndcg@10"]),
        "delta_final_minus_stage2@2": _mean(metrics["delta_final_minus_stage2@2"]),
        "delta_final_minus_stage2@5": _mean(metrics["delta_final_minus_stage2@5"]),
        "delta_final_minus_stage2@10": _mean(metrics["delta_final_minus_stage2@10"]),
        "count": total,
        "stage1_available": _safe_ratio(stats["stage1_available_count"], total),
        "stage2_available": _safe_ratio(stats["stage2_available_count"], total),
        "fallback_used_ratio": _safe_ratio(stats["fallback_used_count"], total),
    }
    return aggregated


def _load_gold_data(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Gold file must contain a list: {path}")
    return {str(item["_id"]): item for item in rows if isinstance(item, dict) and "_id" in item}


def _run_label(pred_path: Path) -> str:
    stem = pred_path.stem
    return re.sub(r"^pred_", "", stem)


def _iter_pred_files(base_dir: Path, pattern: str) -> Iterable[Path]:
    files = sorted(base_dir.glob(pattern))
    for path in files:
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA with stage-wise retrieval recall.")
    parser.add_argument(
        "--base_dir",
        default="/home/wjk/workplace/nq/ano-rag/result/experiment_22",
        help="Directory containing pred_*.jsonl files",
    )
    parser.add_argument(
        "--gold_file",
        default="/home/wjk/workplace/nq/ano-rag/data/hotpot_dev_distractor_500.json",
        help="Gold HotpotQA file (.json or .jsonl)",
    )
    parser.add_argument(
        "--pred_glob",
        default="pred_*.jsonl",
        help="Glob pattern under base_dir",
    )
    parser.add_argument(
        "--report",
        default="evaluation_report_zh.md",
        help="Markdown report filename under base_dir",
    )
    parser.add_argument(
        "--metrics_json",
        default="evaluation_metrics_stagewise.json",
        help="JSON metrics filename under base_dir",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    gold_file = Path(args.gold_file).resolve()
    report_path = base_dir / args.report
    metrics_path = base_dir / args.metrics_json

    print(f"Loading gold data from {gold_file} ...")
    gold_data = _load_gold_data(gold_file)
    print(f"Gold loaded: {len(gold_data)} samples")

    pred_files = list(_iter_pred_files(base_dir, args.pred_glob))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {base_dir} with pattern {args.pred_glob}")

    results: Dict[str, Dict[str, float]] = {}
    for pred_file in pred_files:
        label = _run_label(pred_file)
        print(f"Evaluating {label} ...")
        results[label] = evaluate_run(pred_file, gold_data)

    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# HotpotQA 分阶段评估报告\n\n")
        handle.write("## 1. 评估设置\n")
        handle.write(f"- **评估目录**: `{base_dir}`\n")
        handle.write(f"- **预测文件模式**: `{args.pred_glob}`\n")
        handle.write(f"- **金标文件**: `{gold_file}`\n")
        handle.write(f"- **样本数**: {len(gold_data)}\n\n")

        handle.write("## 2. QA + Final Retrieval\n\n")
        handle.write("| 运行 | EM | F1 | Final R@2 | Final R@5 | Final R@10 | Final NDCG@10 |\n")
        handle.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for label in sorted(results.keys()):
            row = results[label]
            handle.write(
                f"| `{label}` | {row['em']:.4f} | {row['f1']:.4f} | "
                f"{row['final_recall@2']:.4f} | {row['final_recall@5']:.4f} | "
                f"{row['final_recall@10']:.4f} | {row['final_ndcg@10']:.4f} |\n"
            )

        handle.write("\n## 3. Stage-wise Retrieval (公平口径)\n\n")
        handle.write(
            "| 运行 | Stage1 Avail | Stage1 R@10 | Stage2 Avail | Stage2 R@10 | "
            "Final-Stage2 ΔR@10 | Fallback Used |\n"
        )
        handle.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for label in sorted(results.keys()):
            row = results[label]
            handle.write(
                f"| `{label}` | {row['stage1_available']:.4f} | {row['stage1_recall@10']:.4f} | "
                f"{row['stage2_available']:.4f} | {row['stage2_recall@10']:.4f} | "
                f"{row['delta_final_minus_stage2@10']:.4f} | {row['fallback_used_ratio']:.4f} |\n"
            )

        handle.write("\n## 4. 说明\n")
        handle.write("- `Stage1` 使用 `retrieval_stages.stage1.contexts_topk`（hybrid pre-candidates）。\n")
        handle.write("- `Stage2` 使用 `retrieval_stages.stage2_no_fallback.contexts_topk`（hybrid final，不含fallback补底）。\n")
        handle.write("- `Final` 使用 `retrieval_stages.final_with_fallback.contexts_topk`，若缺失则回退到 `retrieved_context_topk`。\n")
        handle.write("- 旧结果若未记录 `retrieval_stages`，`Stage1/Stage2` 可用率会低。\n")

    print(f"Stage-wise metrics saved to {metrics_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
