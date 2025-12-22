#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.output_protocol import extract_final_answer, has_final_tag, normalize_text


def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization for EM/F1."""
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text or ""))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(gold_tokens))
    return (2 * precision * recall) / max(1e-12, precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def _metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths) if ground_truths else 0.0


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_hotpotqa(dataset_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    answers: Dict[str, List[str]] = {}
    support_titles: Dict[str, List[str]] = {}
    for item in data:
        sid = str(item.get("id") or item.get("_id") or "")
        if not sid:
            continue
        ans = item.get("answer")
        answers[sid] = [str(ans)] if isinstance(ans, str) else [str(a) for a in (ans or [])]
        supp = item.get("supporting_facts") or {}
        titles = supp.get("title") or []
        support_titles[sid] = [str(t) for t in titles if str(t)]
    return answers, support_titles


def _load_musique(dataset_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    answers: Dict[str, List[str]] = {}
    support_pids: Dict[str, List[str]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("id") or item.get("query_id") or "")
            if not sid:
                continue
            gold = []
            if item.get("answer"):
                gold.append(str(item["answer"]))
            gold.extend([str(a) for a in item.get("answer_aliases") or [] if str(a)])
            answers[sid] = list(dict.fromkeys(gold))

            paragraphs = item.get("paragraphs") or []
            pids: List[str] = []
            for idx, p in enumerate(paragraphs):
                if not isinstance(p, dict):
                    continue
                if not p.get("is_supporting"):
                    continue
                pid = str(p.get("pid") or p.get("para_id") or p.get("id") or f"p{idx:04d}")
                pids.append(pid)
            support_pids[sid] = pids
    return answers, support_pids


def _load_mirage(dataset_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    answers: Dict[str, List[str]] = {}
    gold_docs: Dict[str, List[str]] = {}
    for item in data:
        sid = str(item.get("query_id") or item.get("id") or "")
        if not sid:
            continue
        ans = item.get("answer")
        if isinstance(ans, list):
            answers[sid] = [str(a) for a in ans]
        elif ans is not None:
            answers[sid] = [str(ans)]
        else:
            answers[sid] = []
        doc_name = str(item.get("doc_name") or item.get("doc_id") or "")
        gold_docs[sid] = [doc_name] if doc_name else []
    return answers, gold_docs


def _norm_doc(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.translate(str.maketrans({c: " " for c in string.punctuation}))
    return " ".join(cleaned.split())


def _sort_retrieved(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    if "rank" in items[0]:
        return sorted(items, key=lambda x: int(x.get("rank") or 0))
    return sorted(items, key=lambda x: float(x.get("score") or 0.0), reverse=True)


def _evaluate_retrieval(
    records: List[Dict[str, Any]],
    gold_map: Dict[str, List[str]],
    *,
    ks: Sequence[int],
    use_passage_id: bool = False,
    doc_match: bool = False,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"Hit@{k}"] = 0.0
        metrics[f"MultiHop@{k}"] = 0.0
    total = 0
    for rec in records:
        sid = str(rec.get("id") or "")
        gold = gold_map.get(sid) or []
        retrieved = rec.get("retrieved") or []
        retrieved = _sort_retrieved(retrieved)
        if not gold:
            continue
        total += 1
        if use_passage_id:
            keys = [str(it.get("passage_id") or it.get("doc_id") or it.get("title") or "") for it in retrieved]
        else:
            keys = [str(it.get("title") or it.get("doc_id") or it.get("passage_id") or "") for it in retrieved]
        for k in ks:
            topk = keys[:k]
            if doc_match:
                gold_norm = _norm_doc(gold[0]) if gold else ""
                matched = [t for t in topk if gold_norm and gold_norm in _norm_doc(t)]
                inter = len(matched)
                recall = 1.0 if inter > 0 else 0.0
                hit = 1.0 if inter > 0 else 0.0
                multihop = recall
            else:
                gold_set = set(gold)
                inter = len(set(topk) & gold_set)
                recall = inter / max(1, len(gold_set))
                hit = 1.0 if inter > 0 else 0.0
                multihop = 1.0 if inter == len(gold_set) else 0.0
            metrics[f"Recall@{k}"] += recall
            metrics[f"Hit@{k}"] += hit
            metrics[f"MultiHop@{k}"] += multihop
    if total == 0:
        return {k: 0.0 for k in metrics}
    return {k: v / total for k, v in metrics.items()}


def _budget_violation_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0
    violations = 0
    for row in rows:
        budget = row.get("context_budget_tokens")
        used = row.get("context_tokens_used")
        if budget is None or used is None:
            continue
        try:
            total += 1
            if int(used) > int(budget):
                violations += 1
        except Exception:
            continue
    return violations / total if total else 0.0


def _load_pred_raw(pred_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_jsonl(pred_path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("id") or row.get("query_id") or row.get("_id") or "")
        if not sid:
            continue
        out[sid] = row
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_run_dirs(root: Path) -> List[Path]:
    if (root / "preds" / "pred_raw.jsonl").exists():
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir()])


def evaluate_run(
    run_dir: Path,
    dataset_name: str,
    answers_map: Dict[str, List[str]],
    support_map: Dict[str, List[str]] | Dict[str, str],
    *,
    ks: Sequence[int],
    max_final_tokens: int,
) -> Dict[str, Any]:
    preds_dir = run_dir / "preds"
    metrics_dir = run_dir / "metrics"
    pred_raw_path = preds_dir / "pred_raw.jsonl"
    retrieval_path = run_dir / "artifacts" / "retrieval.jsonl"

    pred_raw = _load_pred_raw(pred_raw_path)
    pred_norm_rows: List[Dict[str, Any]] = []
    pred_final_rows: List[Dict[str, Any]] = []
    raw_texts: List[str] = []

    em_norm = f1_norm = em_final = f1_final = 0.0
    count = 0

    for sid, row in pred_raw.items():
        raw = str(row.get("pred_raw") or "")
        raw_texts.append(raw)
        pred_norm = normalize_text(raw, lowercase=True, normalize_punct=True)
        pred_final = extract_final_answer(raw, max_tokens=max_final_tokens)
        pred_norm_rows.append({"id": sid, "pred_norm": pred_norm})
        pred_final_rows.append({"id": sid, "pred_final": pred_final})

        golds = answers_map.get(sid)
        if not golds:
            continue
        count += 1
        em_norm += _metric_max_over_ground_truths(_exact_match, pred_norm, golds)
        f1_norm += _metric_max_over_ground_truths(_f1_score, pred_norm, golds)
        em_final += _metric_max_over_ground_truths(_exact_match, pred_final, golds)
        f1_final += _metric_max_over_ground_truths(_f1_score, pred_final, golds)

    if count == 0:
        qa_norm = {"EM": 0.0, "F1": 0.0, "count": 0}
        qa_final = {"EM": 0.0, "F1": 0.0, "count": 0}
    else:
        qa_norm = {"EM": em_norm / count, "F1": f1_norm / count, "count": count}
        qa_final = {"EM": em_final / count, "F1": f1_final / count, "count": count}

    invalid = sum(1 for raw in raw_texts if not extract_final_answer(raw, max_tokens=max_final_tokens))
    missing_final = sum(1 for raw in raw_texts if not has_final_tag(raw))
    total_raw = max(1, len(raw_texts))
    format_metrics = {
        "invalid_rate": invalid / total_raw,
        "no_final_tag_rate": missing_final / total_raw,
        "count": len(raw_texts),
        "budget_violation_rate": _budget_violation_rate(list(pred_raw.values())),
    }

    _write_jsonl(preds_dir / "pred_norm.jsonl", pred_norm_rows)
    _write_jsonl(preds_dir / "pred_final.jsonl", pred_final_rows)
    _write_json(metrics_dir / "qa_metrics_norm.json", qa_norm)
    _write_json(metrics_dir / "qa_metrics_final.json", qa_final)
    _write_json(metrics_dir / "format_metrics.json", format_metrics)

    retrieval_metrics: Dict[str, float] = {}
    if retrieval_path.exists():
        records = _load_jsonl(retrieval_path)
        if dataset_name == "hotpotqa":
            retrieval_metrics = _evaluate_retrieval(records, support_map, ks=ks, use_passage_id=False)
        elif dataset_name == "musique":
            retrieval_metrics = _evaluate_retrieval(records, support_map, ks=ks, use_passage_id=True)
        elif dataset_name == "mirage":
            retrieval_metrics = _evaluate_retrieval(records, support_map, ks=ks, use_passage_id=False, doc_match=True)
    _write_json(metrics_dir / "retrieval_metrics.json", retrieval_metrics)

    summary = {
        "run_dir": str(run_dir),
        "qa_norm": qa_norm,
        "qa_final": qa_final,
        "format_metrics": format_metrics,
        "retrieval_metrics": retrieval_metrics,
        "pred_raw": str(pred_raw_path),
        "pred_norm": str(preds_dir / "pred_norm.jsonl"),
        "pred_final": str(preds_dir / "pred_final.jsonl"),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RelRAG evaluation (protocol + retrieval).")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--dataset-name", required=True, choices=["hotpotqa", "musique", "mirage"])
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Single run directory")
    parser.add_argument("--root", default=None, help="Root containing multiple run dirs")
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--max-final-tokens", type=int, default=50)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.dataset_name == "hotpotqa":
        answers_map, support_map = _load_hotpotqa(dataset_path)
    elif args.dataset_name == "musique":
        answers_map, support_map = _load_musique(dataset_path)
    else:
        answers_map, support_map = _load_mirage(dataset_path)

    if args.work_dir:
        run_dirs = [Path(args.work_dir)]
    else:
        root = Path(args.root or "result_relrag")
        run_dirs = _collect_run_dirs(root)

    ks = [int(k.strip()) for k in str(args.ks).split(",") if k.strip()]
    summaries = []
    for run_dir in run_dirs:
        pred_raw = run_dir / "preds" / "pred_raw.jsonl"
        if not pred_raw.exists():
            continue
        summaries.append(
            evaluate_run(
                run_dir,
                args.dataset_name,
                answers_map,
                support_map,
                ks=ks,
                max_final_tokens=args.max_final_tokens,
            )
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
