#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_answer(text: str) -> str:
    import re
    import string

    def _remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def _white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def _remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def _lower(value: str) -> str:
        return value.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(text))))


def _f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    norm_pred = _normalize_answer(prediction)
    norm_gold = _normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)
    if norm_pred in {"yes", "no", "noanswer"} and norm_pred != norm_gold:
        return zero
    if norm_gold in {"yes", "no", "noanswer"} and norm_pred != norm_gold:
        return zero
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return zero
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def _update_sp(prediction: List[List[Any]], gold: List[List[Any]]) -> Tuple[float, float, float, float]:
    pred_set = set(tuple(item) for item in prediction)
    gold_set = set(tuple(item) for item in gold)
    tp = len([item for item in pred_set if item in gold_set])
    fp = len([item for item in pred_set if item not in gold_set])
    fn = len([item for item in gold_set if item not in pred_set])
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * prec * recall) / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1, prec, recall


def _evaluate_official(pred_path: Path, gold_path: Path) -> Optional[Dict[str, float]]:
    if not pred_path.exists() or not gold_path.exists():
        return None
    with pred_path.open("r", encoding="utf-8") as handle:
        prediction = json.load(handle)
    with gold_path.open("r", encoding="utf-8") as handle:
        gold = json.load(handle)

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
    for dp in gold:
        raw_id = dp.get("_id")
        if raw_id is None:
            continue
        cur_id = str(raw_id)
        if cur_id is None:
            continue
        can_eval_joint = True
        pred_ans = prediction.get("answer", {}).get(cur_id)
        if pred_ans is None:
            can_eval_joint = False
        else:
            em = 1.0 if _normalize_answer(pred_ans) == _normalize_answer(dp.get("answer", "")) else 0.0
            f1, prec, recall = _f1_score(pred_ans, dp.get("answer", ""))
            metrics["em"] += em
            metrics["f1"] += f1
            metrics["prec"] += prec
            metrics["recall"] += recall

        pred_sp = prediction.get("sp", {}).get(cur_id)
        if pred_sp is None:
            can_eval_joint = False
        else:
            sp_em, sp_f1, sp_prec, sp_recall = _update_sp(pred_sp, dp.get("supporting_facts", []))
            metrics["sp_em"] += sp_em
            metrics["sp_f1"] += sp_f1
            metrics["sp_prec"] += sp_prec
            metrics["sp_recall"] += sp_recall

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            joint_f1 = (2 * joint_prec * joint_recall / (joint_prec + joint_recall)) if joint_prec + joint_recall > 0 else 0.0
            joint_em = em * sp_em
            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    total = len(gold) or 1
    for key in metrics:
        metrics[key] = metrics[key] / total
    return metrics


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _topk_list(record: Dict[str, Any]) -> Tuple[List[Any], Optional[str]]:
    for key in ("retrieved_context_topk", "retrieved_topk", "retrieved_context"):
        if key in record and isinstance(record.get(key), list):
            return record.get(key) or [], key
    return [], None


def _analyze_pred(path: Path) -> Dict[str, Any]:
    required_fields = [
        "retrieved_context_raw",
        "retrieved_context_topk",
        "top_k",
        "top_k_raw",
        "top_k_final",
        "duplicate_rate",
        "overfetch_factor",
        "answer_source",
        "fallback_reason",
        "llm_error",
    ]
    missing_counts = {key: 0 for key in required_fields}
    total = 0
    top_k_final_match = 0
    top_k_final_total = 0
    duplicate_rates: List[float] = []
    overfetch_factors: List[float] = []
    fallback_count = 0
    topk_empty = 0
    topk_field_used: Dict[str, int] = {}
    for record in _load_jsonl(path):
        total += 1
        for key in required_fields:
            if key not in record:
                missing_counts[key] += 1
        top_k = record.get("top_k") or (record.get("intermediate") or {}).get("top_k")
        top_k_final = record.get("top_k_final")
        if isinstance(top_k, (int, float)) and isinstance(top_k_final, (int, float)):
            top_k_final_total += 1
            if int(top_k_final) == int(top_k):
                top_k_final_match += 1
        duplicate_rate = record.get("duplicate_rate")
        if isinstance(duplicate_rate, (int, float)):
            duplicate_rates.append(float(duplicate_rate))
        overfetch = record.get("overfetch_factor")
        if not isinstance(overfetch, (int, float)):
            top_k_raw = record.get("top_k_raw")
            if isinstance(top_k_raw, (int, float)) and isinstance(top_k, (int, float)) and top_k:
                overfetch = float(top_k_raw) / float(top_k)
        if isinstance(overfetch, (int, float)):
            overfetch_factors.append(float(overfetch))
        if record.get("answer_source") == "llm_fallback":
            fallback_count += 1
        topk_list, used_key = _topk_list(record)
        if used_key:
            topk_field_used[used_key] = topk_field_used.get(used_key, 0) + 1
        if not topk_list:
            topk_empty += 1
    return {
        "total": total,
        "missing_counts": missing_counts,
        "top_k_final_match": top_k_final_match,
        "top_k_final_total": top_k_final_total,
        "top_k_final_coverage": (top_k_final_match / top_k_final_total) if top_k_final_total else 0.0,
        "duplicate_rate_mean": _safe_mean(duplicate_rates),
        "overfetch_factor_mean": _safe_mean(overfetch_factors),
        "fallback_ratio": (fallback_count / total) if total else 0.0,
        "topk_empty_ratio": (topk_empty / total) if total else 0.0,
        "topk_field_used": topk_field_used,
    }


def _pair_set(items: Any) -> set[Tuple[Any, Any]]:
    pairs = set()
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            pairs.add((str(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
    return pairs


def _analyze_alignment_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    total = 0
    pred_eq_gold = 0
    pred_eq_topk = 0
    gold_subset = 0
    for record in _load_jsonl(path):
        total += 1
        gold_sp = _pair_set(record.get("gold_sp"))
        pred_sp = _pair_set(record.get("pred_sp"))
        topk_sp = _pair_set(record.get("retrieved_context_topk"))
        if gold_sp and pred_sp == gold_sp:
            pred_eq_gold += 1
        if pred_sp == topk_sp:
            pred_eq_topk += 1
        if gold_sp.issubset(topk_sp):
            gold_subset += 1
    if total == 0:
        return None
    return {
        "total": total,
        "pred_eq_gold_ratio": pred_eq_gold / total,
        "pred_eq_topk_ratio": pred_eq_topk / total,
        "gold_subset_ratio": gold_subset / total,
    }


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _collect_pred_files(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("pred_*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline acceptance verifier for Hotpot runs")
    parser.add_argument("--run_root", required=True, help="Run root directory containing pred_*.jsonl")
    parser.add_argument("--gold_jsonl", required=True, help="Gold JSONL path used for the run")
    parser.add_argument("--official_gold", required=True, help="Official gold JSON path for official eval")
    parser.add_argument("--out", help="Output markdown report path (default: run_root/acceptance_report.md)")
    parser.add_argument("--out_json", help="Output JSON report path (default: run_root/acceptance_report.json)")
    parser.add_argument("--top_k_final_target", type=float, default=0.99, help="Target coverage for top_k_final==top_k")
    parser.add_argument("--dup_rate_max", type=float, default=0.10, help="Max mean duplicate_rate")
    parser.add_argument("--overfetch_min", type=float, default=1.8, help="Min mean overfetch_factor")
    parser.add_argument("--overfetch_max", type=float, default=4.0, help="Max mean overfetch_factor")
    parser.add_argument("--fallback_ratio_max", type=float, default=0.01, help="Max llm_fallback ratio")
    parser.add_argument("--gold_subset_min", type=float, default=0.95, help="Min gold_sp subset ratio")
    parser.add_argument("--sp_recall_min", type=float, default=0.5, help="Min official sp_recall")
    parser.add_argument("--joint_recall_min", type=float, default=0.3, help="Min official joint_recall")
    parser.add_argument("--empty_topk_issue_ratio", type=float, default=0.95, help="Ratio of empty retrieved_topk to flag data issue")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    pred_files = _collect_pred_files(run_root)
    if not pred_files:
        raise SystemExit(f"No pred_*.jsonl found under {run_root}")

    out_path = Path(args.out) if args.out else run_root / "acceptance_report.md"
    out_json_path = Path(args.out_json) if args.out_json else run_root / "acceptance_report.json"
    gold_official = Path(args.official_gold)

    thresholds = {
        "top_k_final_target": args.top_k_final_target,
        "dup_rate_max": args.dup_rate_max,
        "overfetch_min": args.overfetch_min,
        "overfetch_max": args.overfetch_max,
        "fallback_ratio_max": args.fallback_ratio_max,
        "gold_subset_min": args.gold_subset_min,
        "sp_recall_min": args.sp_recall_min,
        "joint_recall_min": args.joint_recall_min,
        "empty_topk_issue_ratio": args.empty_topk_issue_ratio,
    }

    runs_report: List[Dict[str, Any]] = []
    for pred_path in pred_files:
        align_dir = run_root / f"align_{pred_path.stem}"
        snapshot_path = align_dir / "alignment_audit_snapshot.jsonl"
        official_pred = align_dir / "official_pred.json"
        official_metrics = _evaluate_official(official_pred, gold_official)
        pred_stats = _analyze_pred(pred_path)
        align_stats = _analyze_alignment_snapshot(snapshot_path)

        reasons: List[str] = []
        data_issue = False
        if pred_stats["topk_empty_ratio"] >= args.empty_topk_issue_ratio and official_metrics:
            if official_metrics.get("sp_recall", 0.0) > 0.1 or official_metrics.get("joint_recall", 0.0) > 0.1:
                data_issue = True
                reasons.append("Data-Issue Fail: retrieved_topk empty but official recall > 0.1")

        p0_ok = True
        if pred_stats["top_k_final_coverage"] < args.top_k_final_target:
            p0_ok = False
            reasons.append("P0: top_k_final coverage below target")
        if pred_stats["duplicate_rate_mean"] > args.dup_rate_max:
            p0_ok = False
            reasons.append("P0: duplicate_rate above threshold")
        if pred_stats["overfetch_factor_mean"] < args.overfetch_min:
            p0_ok = False
            reasons.append("P0: overfetch_factor below minimum")
        if pred_stats["overfetch_factor_mean"] > args.overfetch_max:
            p0_ok = False
            reasons.append("P0: overfetch_factor above maximum")

        p1_status = "N/A"
        p1_ok = None
        if align_stats:
            p1_ok = align_stats["gold_subset_ratio"] >= args.gold_subset_min
            p1_status = "PASS" if p1_ok else "FAIL"
            if not p1_ok:
                reasons.append("P1: gold_sp subset ratio below target")

        output_ok = pred_stats["fallback_ratio"] <= args.fallback_ratio_max
        if not output_ok:
            reasons.append("Output: llm_fallback ratio above threshold")

        final_ok = None
        final_status = "N/A"
        if official_metrics:
            final_ok = (
                official_metrics.get("sp_recall", 0.0) >= args.sp_recall_min
                and official_metrics.get("joint_recall", 0.0) >= args.joint_recall_min
            )
            final_status = "PASS" if final_ok else "FAIL"
            if not final_ok:
                reasons.append("Final: official recall metrics below threshold")

        top_reasons = []
        for reason in reasons:
            if reason not in top_reasons:
                top_reasons.append(reason)
            if len(top_reasons) >= 3:
                break

        runs_report.append(
            {
                "run": pred_path.stem,
                "pred_path": str(pred_path),
                "align_dir": str(align_dir),
                "snapshot_path": str(snapshot_path) if snapshot_path.exists() else None,
                "p0": "PASS" if p0_ok else "FAIL",
                "p1": p1_status,
                "output_layer": "PASS" if output_ok else "FAIL",
                "final_metrics": final_status,
                "data_issue": "FAIL" if data_issue else "PASS",
                "top_reasons": top_reasons,
                "pred_stats": pred_stats,
                "alignment_stats": align_stats,
                "official_metrics": official_metrics,
            }
        )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_root": str(run_root),
        "thresholds": thresholds,
        "runs": runs_report,
    }
    out_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Acceptance Report")
    lines.append("")
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- run_root: {report['run_root']}")
    lines.append("- thresholds:")
    for key, value in thresholds.items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    for run in runs_report:
        lines.append(f"## {run['run']}")
        lines.append(f"- P0: {run['p0']}")
        lines.append(f"- P1: {run['p1']}")
        lines.append(f"- Output Layer: {run['output_layer']}")
        lines.append(f"- Final Metrics: {run['final_metrics']}")
        lines.append(f"- Data Issue: {run['data_issue']}")
        if run["top_reasons"]:
            lines.append(f"- Top Reasons: {', '.join(run['top_reasons'])}")
        pred_stats = run["pred_stats"]
        lines.append(
            "- pred_stats:"
            f" top_k_final_coverage={_format_ratio(pred_stats['top_k_final_coverage'])}"
            f" duplicate_rate_mean={pred_stats['duplicate_rate_mean']:.4f}"
            f" overfetch_factor_mean={pred_stats['overfetch_factor_mean']:.4f}"
            f" fallback_ratio={pred_stats['fallback_ratio']:.4f}"
            f" topk_empty_ratio={pred_stats['topk_empty_ratio']:.4f}"
        )
        align_stats = run.get("alignment_stats")
        if align_stats:
            lines.append(
                "- alignment_stats:"
                f" gold_subset_ratio={_format_ratio(align_stats.get('gold_subset_ratio'))}"
                f" pred_eq_topk_ratio={_format_ratio(align_stats.get('pred_eq_topk_ratio'))}"
                f" pred_eq_gold_ratio={_format_ratio(align_stats.get('pred_eq_gold_ratio'))}"
            )
        else:
            lines.append("- alignment_stats: N/A (alignment_audit_snapshot.jsonl missing)")
        official_metrics = run.get("official_metrics")
        if official_metrics:
            lines.append(
                "- official_metrics:"
                f" sp_recall={_format_ratio(official_metrics.get('sp_recall'))}"
                f" joint_recall={_format_ratio(official_metrics.get('joint_recall'))}"
                f" f1={_format_ratio(official_metrics.get('f1'))}"
            )
        else:
            lines.append("- official_metrics: N/A (official_pred or official_gold missing)")
        missing_counts = run["pred_stats"]["missing_counts"]
        missing_keys = [k for k, v in missing_counts.items() if v > 0]
        if missing_keys:
            lines.append(f"- missing_fields: {', '.join(missing_keys)}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_json_path}")


if __name__ == "__main__":
    main()
