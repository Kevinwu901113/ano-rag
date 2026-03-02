#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from eval_retrieval_v3 import evaluate_retrieval


BASELINE_METHODS = ("bm25", "dense", "lightrag", "graphrag", "raptor")
DATASETS = ("hotpotqa", "2wiki", "musique")
BACKENDS = ("qwen", "deepseek")


def _norm_answer(text: Any) -> str:
    value = str(text or "").lower()
    exclude = set(string.punctuation)
    value = "".join(ch for ch in value if ch not in exclude)
    tokens = [tok for tok in value.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def _f1(prediction: str, gold: str) -> float:
    pred = _norm_answer(prediction)
    gt = _norm_answer(gold)
    if pred in {"yes", "no", "noanswer"} and pred != gt:
        return 0.0
    if gt in {"yes", "no", "noanswer"} and pred != gt:
        return 0.0
    pred_tokens = pred.split()
    gold_tokens = gt.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same <= 0:
        return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def _em(prediction: str, gold: str) -> float:
    return 1.0 if _norm_answer(prediction) == _norm_answer(gold) else 0.0


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_gold_map(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in _iter_jsonl(path):
        qid = str(row.get("_id") or row.get("id") or "").strip()
        if not qid:
            continue
        answers: List[str] = []
        answer = row.get("answer")
        if isinstance(answer, list):
            answers.extend(str(x).strip() for x in answer if str(x).strip())
        elif answer is not None:
            answers.append(str(answer).strip())
        aliases = row.get("answer_aliases")
        if isinstance(aliases, list):
            answers.extend(str(x).strip() for x in aliases if str(x).strip())
        if not answers:
            answers = [""]
        out[qid] = answers
    return out


def compute_qa_metrics(pred_file: Path, gold_file: Path) -> Dict[str, Any]:
    gold_map = _load_gold_map(gold_file)
    em_vals: List[float] = []
    f1_vals: List[float] = []
    for row in _iter_jsonl(pred_file):
        qid = str(row.get("_id") or row.get("id") or "").strip()
        if not qid or qid not in gold_map:
            continue
        pred = row.get("short_answer", row.get("prediction", row.get("pred", row.get("answer", ""))))
        pred_text = str(pred or "")
        refs = gold_map[qid]
        em_vals.append(max(_em(pred_text, ref) for ref in refs))
        f1_vals.append(max(_f1(pred_text, ref) for ref in refs))

    count = len(em_vals)
    return {
        "em": (sum(em_vals) / count) if count else 0.0,
        "f1": (sum(f1_vals) / count) if count else 0.0,
        "count": count,
    }


def _summarize_cost_rows(rows: List[Dict[str, Any]], *, method: str, dataset: str, backend: str) -> Dict[str, Any]:
    def collect(key: str) -> List[float]:
        vals: List[float] = []
        for row in rows:
            cost = row.get("cost") if isinstance(row, dict) else None
            if not isinstance(cost, dict):
                continue
            value = cost.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return vals

    def summary(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0, "sum": None, "avg": None, "p50": None, "p95": None}
        ordered = sorted(vals)
        n = len(ordered)
        p50 = ordered[int(0.50 * (n - 1))]
        p95 = ordered[int(0.95 * (n - 1))]
        return {
            "count": n,
            "sum": round(sum(ordered), 3),
            "avg": round(sum(ordered) / n, 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
        }

    token_source_counts: Dict[str, int] = {}
    missing_cost_count = 0
    for row in rows:
        cost = row.get("cost") if isinstance(row, dict) else None
        if not isinstance(cost, dict):
            missing_cost_count += 1
            continue
        src = str(cost.get("token_source") or "unknown")
        token_source_counts[src] = token_source_counts.get(src, 0) + 1

    return {
        "method": method,
        "dataset": dataset,
        "backend": backend,
        "count": len(rows),
        "missing_cost_count": missing_cost_count,
        "token_source_counts": token_source_counts,
        "index_time_ms": summary(collect("index_time_ms")),
        "query_time_ms": summary(collect("query_time_ms")),
        "query_retrieval_ms": summary(collect("query_retrieval_ms")),
        "query_reader_ms": summary(collect("query_reader_ms")),
        "prompt_tokens_total": summary(collect("prompt_tokens_total")),
        "completion_tokens_total": summary(collect("completion_tokens_total")),
        "total_tokens": summary(collect("total_tokens")),
        "llm_calls": summary(collect("llm_calls")),
        "llm_retries": summary(collect("llm_retries")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _discover_runs(baseline_root: Path, relrag_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for method in BASELINE_METHODS:
        for dataset in DATASETS:
            for backend in BACKENDS:
                run_dir = baseline_root / method / dataset / backend
                pred = run_dir / "pred.jsonl"
                retrieval_pred = run_dir / "pred_retrieval.jsonl"
                if not pred.exists() and not retrieval_pred.exists():
                    continue
                runs.append(
                    {
                        "family": "baseline",
                        "method": method,
                        "dataset": dataset,
                        "backend": backend,
                        "run_dir": run_dir,
                        "pred_file": pred if pred.exists() else retrieval_pred,
                        "retrieval_pred_file": retrieval_pred if retrieval_pred.exists() else (pred if pred.exists() else None),
                    }
                )

    for dataset in DATASETS:
        for backend in BACKENDS:
            run_dir = relrag_root / dataset / backend
            pred_candidates = [
                run_dir / "predictions.jsonl",
                run_dir / "pred_dev_hybrid.jsonl",
                run_dir / "pred_dev_vllm_hybrid.jsonl",
                run_dir / "pred_dev_openai_hybrid.jsonl",
            ]
            pred = None
            for candidate in pred_candidates:
                if candidate.exists():
                    pred = candidate
                    break
            if pred is None:
                dynamic = sorted(
                    p for p in run_dir.glob("pred*.jsonl")
                    if p.name not in {"retrieved_context_raw.jsonl", "retrieved_context_topk.jsonl"}
                )
                if dynamic:
                    pred = dynamic[0]
            if pred is None:
                continue
            runs.append(
                {
                    "family": "relrag",
                    "method": "hybrid",
                    "dataset": dataset,
                    "backend": backend,
                    "run_dir": run_dir,
                    "pred_file": pred,
                    "retrieval_pred_file": pred,
                }
            )
    return runs


def _gold_for_dataset(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "hotpotqa":
        return Path(args.gold_hotpot)
    if dataset == "2wiki":
        return Path(args.gold_2wiki)
    if dataset == "musique":
        return Path(args.gold_musique)
    raise ValueError(f"Unknown dataset: {dataset}")


def _to_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate aligned baseline/relrag metrics and costs.")
    parser.add_argument("--baseline_root", default="baseline/results")
    parser.add_argument("--relrag_root", default="result/aligned/relrag/hybrid")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gold_hotpot", default="data/hotpot_dev_fullwiki_500_jsonl.jsonl")
    parser.add_argument("--gold_2wiki", default="data/2wiki_dev_sample_500.jsonl")
    parser.add_argument("--gold_musique", default="data/musique_ans_v1.0_dev_500.jsonl")
    parser.add_argument("--expected_count", type=int, default=500)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    relrag_root = Path(args.relrag_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(baseline_root, relrag_root)
    metrics_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []

    for run in runs:
        dataset = str(run["dataset"])
        gold_file = _gold_for_dataset(dataset, args)
        pred_file = Path(run["pred_file"])
        retrieval_pred = Path(run["retrieval_pred_file"]) if run.get("retrieval_pred_file") else pred_file
        run_dir = Path(run["run_dir"])

        qa_metrics = compute_qa_metrics(pred_file, gold_file)
        retrieval_metrics = evaluate_retrieval(
            pred_file=retrieval_pred,
            gold_file=gold_file,
            expected_count=int(args.expected_count),
        )

        pred_rows = list(_iter_jsonl(pred_file))
        cost_summary = _summarize_cost_rows(
            pred_rows,
            method=str(run["method"]),
            dataset=dataset,
            backend=str(run["backend"]),
        )

        metrics_all = {
            "family": run["family"],
            "method": run["method"],
            "dataset": dataset,
            "backend": run["backend"],
            "metrics_qa": qa_metrics,
            "metrics_retrieval": retrieval_metrics,
            "cost_summary": cost_summary,
            "paths": {
                "pred_file": str(pred_file),
                "retrieval_pred_file": str(retrieval_pred),
                "gold_file": str(gold_file),
            },
        }

        _write_json(run_dir / "metrics_qa.json", qa_metrics)
        _write_json(run_dir / "metrics_retrieval.json", retrieval_metrics)
        _write_json(run_dir / "cost_summary.json", cost_summary)
        _write_json(run_dir / "metrics_all.json", metrics_all)

        metrics_rows.append(
            {
                "family": run["family"],
                "method": run["method"],
                "dataset": dataset,
                "backend": run["backend"],
                "em": round(float(qa_metrics.get("em") or 0.0), 6),
                "f1": round(float(qa_metrics.get("f1") or 0.0), 6),
                "recall@2": _to_float(retrieval_metrics.get("recall@2")),
                "recall@5": _to_float(retrieval_metrics.get("recall@5")),
                "ie@2": _to_float(retrieval_metrics.get("ie@2")),
                "ie@5": _to_float(retrieval_metrics.get("ie@5")),
                "ndcg@2": _to_float(retrieval_metrics.get("ndcg@2")),
                "ndcg@5": _to_float(retrieval_metrics.get("ndcg@5")),
                "count": int(qa_metrics.get("count") or 0),
                "retrieval_mode": retrieval_metrics.get("recall_mode"),
                "alignment_ok": bool(retrieval_metrics.get("alignment_ok")),
            }
        )

        cost_rows.append(
            {
                "family": run["family"],
                "method": run["method"],
                "dataset": dataset,
                "backend": run["backend"],
                "count": int(cost_summary.get("count") or 0),
                "index_time_ms_avg": cost_summary["index_time_ms"].get("avg"),
                "query_time_ms_avg": cost_summary["query_time_ms"].get("avg"),
                "query_retrieval_ms_avg": cost_summary["query_retrieval_ms"].get("avg"),
                "query_reader_ms_avg": cost_summary["query_reader_ms"].get("avg"),
                "total_tokens_avg": cost_summary["total_tokens"].get("avg"),
                "llm_calls_avg": cost_summary["llm_calls"].get("avg"),
                "token_source_counts": json.dumps(cost_summary.get("token_source_counts") or {}, ensure_ascii=False),
            }
        )

        full_rows.append(metrics_all)

    metrics_csv = output_dir / "comparison_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "method",
                "dataset",
                "backend",
                "em",
                "f1",
                "recall@2",
                "recall@5",
                "ie@2",
                "ie@5",
                "ndcg@2",
                "ndcg@5",
                "count",
                "retrieval_mode",
                "alignment_ok",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    cost_csv = output_dir / "comparison_cost.csv"
    with cost_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "method",
                "dataset",
                "backend",
                "count",
                "index_time_ms_avg",
                "query_time_ms_avg",
                "query_retrieval_ms_avg",
                "query_reader_ms_avg",
                "total_tokens_avg",
                "llm_calls_avg",
                "token_source_counts",
            ],
        )
        writer.writeheader()
        for row in cost_rows:
            writer.writerow(row)

    full_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_root": str(baseline_root),
        "relrag_root": str(relrag_root),
        "run_count": len(runs),
        "runs": full_rows,
    }
    _write_json(output_dir / "comparison_full.json", full_payload)

    print(json.dumps({
        "run_count": len(runs),
        "comparison_metrics_csv": str(metrics_csv),
        "comparison_cost_csv": str(cost_csv),
        "comparison_full_json": str(output_dir / "comparison_full.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
