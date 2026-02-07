import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_supporting_facts(raw: Any) -> List[List[Any]]:
    if not raw:
        return []
    normalized: List[List[Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = item[0]
        idx = item[1]
        if title is None:
            continue
        try:
            sent_idx = int(idx)
        except (TypeError, ValueError):
            continue
        normalized.append([str(title), sent_idx])
    return normalized


def _pairs_from_context(context: Any) -> List[List[Any]]:
    pairs: List[List[Any]] = []
    seen = set()
    for item in context or []:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        title = item.get("title")
        idx = item.get("sentence_idx")
        if title is None or idx is None:
            continue
        try:
            sent_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if chunk_id:
            key = ("chunk_id", str(chunk_id))
        else:
            key = ("title_idx", str(title), sent_idx)
        if key in seen:
            continue
        seen.add(key)
        pairs.append([str(title), sent_idx])
    return pairs


def _sp_set(sp: Any) -> set[Tuple[Any, Any]]:
    normalized = _normalize_supporting_facts(sp)
    return set((item[0], item[1]) for item in normalized)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": _mean(values)}


def _ratio(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return num / denom


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hotpot alignment audit + official export")
    parser.add_argument("--pred_jsonl", required=True, help="Prediction JSONL path")
    parser.add_argument("--gold_jsonl", required=True, help="Gold JSONL path")
    parser.add_argument("--output_dir", help="Output directory (default: pred_jsonl parent)")
    parser.add_argument("--pred_official_out", help="Official pred JSON output path")
    parser.add_argument("--pred_official_topk_out", help="Official legacy-topk pred JSON output path")
    parser.add_argument("--gold_official_out", help="Official gold JSON output path")
    parser.add_argument("--split", help="Split label for manifest metadata")
    args = parser.parse_args()

    pred_path = Path(args.pred_jsonl)
    gold_path = Path(args.gold_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_basename = gold_path.stem
    gold_official_out = Path(args.gold_official_out) if args.gold_official_out else output_dir / "official_gold.json"
    pred_official_out = Path(args.pred_official_out) if args.pred_official_out else output_dir / "official_pred.json"
    pred_official_topk_out = (
        Path(args.pred_official_topk_out) if args.pred_official_topk_out else output_dir / "official_pred_topk.json"
    )

    pred_records = list(_load_jsonl(pred_path))
    if not pred_records:
        raise ValueError(f"No prediction records found in {pred_path}")

    gold_records: List[Dict[str, Any]] = []
    for record in _load_jsonl(gold_path):
        qid = record.get("_id")
        if qid is None:
            continue
        answer = record.get("answer") or ""
        supporting_facts = record.get("supporting_facts")
        if supporting_facts is None and "sp" in record:
            supporting_facts = record.get("sp")
        gold_records.append(
            {
                "_id": qid,
                "answer": answer,
                "supporting_facts": _normalize_supporting_facts(supporting_facts),
            }
        )

    _write_json(gold_official_out, gold_records)

    answers: Dict[str, Any] = {}
    supports: Dict[str, Any] = {}
    supports_topk: Dict[str, Any] = {}
    pred_sp_policy_counts: Dict[str, int] = {}
    has_pred_sp_topk = False
    for record in pred_records:
        qid = str(record.get("_id") or "")
        if not qid:
            continue
        if "short_answer" not in record:
            raise ValueError(f"missing short_answer for qid={qid}")
        if "pred_sp" not in record:
            raise ValueError(f"missing pred_sp for qid={qid}")
        pred_sp_topk = record.get("pred_sp_topk")
        if pred_sp_topk is None:
            pred_sp_topk = record.get("pred_sp") or []
        else:
            has_pred_sp_topk = True
        policy = (
            record.get("pred_sp_policy")
            or (record.get("intermediate") or {}).get("pred_sp_policy")
            or "topk"
        )
        policy_key = str(policy)
        pred_sp_policy_counts[policy_key] = pred_sp_policy_counts.get(policy_key, 0) + 1
        answers[qid] = record.get("short_answer") or ""
        supports[qid] = record.get("pred_sp") or []
        supports_topk[qid] = pred_sp_topk or []

    _write_json(pred_official_out, {"answer": answers, "sp": supports})
    if has_pred_sp_topk:
        _write_json(pred_official_topk_out, {"answer": answers, "sp": supports_topk})

    pred_eq_gold = 0
    pred_eq_topk_context = 0
    pred_topk_eq_topk_context = 0
    pred_subset_topk = 0
    gold_subset_topk = 0
    answer_sources: Dict[str, int] = {}
    top_k_values: List[float] = []
    top_k_raw_values: List[float] = []
    top_k_final_values: List[float] = []
    duplicate_rates: List[float] = []
    shortage_refill_added_values: List[float] = []
    shortage_refill_triggered = 0

    snapshot_path = output_dir / "alignment_audit_snapshot.jsonl"
    with snapshot_path.open("w", encoding="utf-8") as snapshot:
        for record in pred_records:
            qid = str(record.get("_id") or "")
            gold_sp = record.get("gold_sp")
            pred_sp = record.get("pred_sp")
            pred_sp_topk = record.get("pred_sp_topk")
            if pred_sp_topk is None:
                pred_sp_topk = pred_sp or []
            if gold_sp is None:
                raise ValueError(f"missing gold_sp for qid={qid}")
            if pred_sp is None:
                raise ValueError(f"missing pred_sp for qid={qid}")

            retrieved_context_raw = record.get("retrieved_context_raw")
            retrieved_context_topk = record.get("retrieved_context_topk")
            if retrieved_context_raw is None:
                raise ValueError(f"missing retrieved_context_raw for qid={qid}")
            if retrieved_context_topk is None:
                raise ValueError(f"missing retrieved_context_topk for qid={qid}")

            raw_pairs = _pairs_from_context(retrieved_context_raw)
            topk_pairs = _pairs_from_context(retrieved_context_topk)
            pred_set = _sp_set(pred_sp)
            pred_topk_set = _sp_set(pred_sp_topk)
            gold_set = _sp_set(gold_sp)
            topk_set = set((item[0], item[1]) for item in topk_pairs)

            if pred_set == gold_set:
                pred_eq_gold += 1
            if pred_set == topk_set:
                pred_eq_topk_context += 1
            if pred_topk_set == topk_set:
                pred_topk_eq_topk_context += 1
            if pred_set.issubset(pred_topk_set):
                pred_subset_topk += 1
            if gold_set.issubset(topk_set):
                gold_subset_topk += 1

            answer_source = record.get("answer_source") or "unknown"
            answer_sources[answer_source] = answer_sources.get(answer_source, 0) + 1
            pred_sp_policy = (
                record.get("pred_sp_policy")
                or (record.get("intermediate") or {}).get("pred_sp_policy")
                or "topk"
            )

            top_k_val = record.get("intermediate", {}).get("top_k")
            if isinstance(top_k_val, (int, float)):
                top_k_values.append(float(top_k_val))
            top_k_raw_val = record.get("top_k_raw")
            if isinstance(top_k_raw_val, (int, float)):
                top_k_raw_values.append(float(top_k_raw_val))
            top_k_final_val = record.get("top_k_final")
            if isinstance(top_k_final_val, (int, float)):
                top_k_final_values.append(float(top_k_final_val))
            duplicate_rate = record.get("duplicate_rate")
            if isinstance(duplicate_rate, (int, float)):
                duplicate_rates.append(float(duplicate_rate))
            shortage_refill = (
                record.get("top_k_shortage_refill")
                or (record.get("intermediate") or {}).get("top_k_shortage_refill")
                or {}
            )
            if isinstance(shortage_refill, dict):
                if shortage_refill.get("triggered"):
                    shortage_refill_triggered += 1
                added = shortage_refill.get("added")
                if isinstance(added, (int, float)):
                    shortage_refill_added_values.append(float(added))

            llm_input_hash = record.get("llm_input_hash") or record.get("intermediate", {}).get("llm_input_hash")
            snapshot_record = {
                "qid": qid,
                "gold_sp": _normalize_supporting_facts(gold_sp),
                "pred_sp": _normalize_supporting_facts(pred_sp),
                "pred_sp_topk": _normalize_supporting_facts(pred_sp_topk),
                "pred_sp_policy": str(pred_sp_policy),
                "retrieved_context_raw": raw_pairs,
                "retrieved_context_topk": topk_pairs,
                "top_k_raw": record.get("top_k_raw"),
                "top_k_final": record.get("top_k_final"),
                "duplicate_rate": record.get("duplicate_rate"),
                "top_k_shortage_refill": shortage_refill,
                "llm_input_hash": llm_input_hash,
                "answer_source": answer_source,
            }
            snapshot.write(json.dumps(snapshot_record, ensure_ascii=False) + "\n")

    total = len(pred_records)
    pred_eq_gold_ratio = _ratio(pred_eq_gold, total)
    pred_eq_topk_context_ratio = _ratio(pred_eq_topk_context, total)
    pred_topk_eq_topk_context_ratio = _ratio(pred_topk_eq_topk_context, total)
    pred_subset_topk_ratio = _ratio(pred_subset_topk, total)
    gold_subset_ratio = _ratio(gold_subset_topk, total)
    pred_sp_policy = "topk"
    if pred_sp_policy_counts:
        pred_sp_policy = max(
            pred_sp_policy_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
    pred_sp_policy_is_mixed = len(pred_sp_policy_counts) > 1

    first_meta = pred_records[0].get("intermediate", {}) if pred_records else {}
    overfetch_factor = None
    if top_k_values and top_k_raw_values:
        top_k_mean = _mean(top_k_values)
        top_k_raw_mean = _mean(top_k_raw_values)
        if top_k_mean > 0:
            overfetch_factor = top_k_raw_mean / top_k_mean

    manifest = {
        "data": {
            "pred_jsonl": str(pred_path),
            "gold_jsonl": str(gold_path),
            "gold_official": str(gold_official_out),
            "split": args.split or "",
            "sample_count": total,
        },
        "run": {
            "reader": pred_records[0].get("reader"),
            "retriever": pred_records[0].get("mode"),
            "model": pred_records[0].get("model"),
            "pred_sp_policy": pred_sp_policy,
            "pred_sp_policy_mixed": pred_sp_policy_is_mixed,
            "pred_sp_policy_distribution": pred_sp_policy_counts,
            "prompt_name": first_meta.get("prompt_name"),
            "prompt_template_hash": first_meta.get("prompt_template_hash"),
            "system_prompt_name": first_meta.get("system_prompt_name"),
            "system_prompt_hash": first_meta.get("system_prompt_hash"),
        },
        "retrieval": {
            "top_k": _summarize(top_k_values),
            "top_k_raw": _summarize(top_k_raw_values),
            "top_k_final": _summarize(top_k_final_values),
            "duplicate_rate": _summarize(duplicate_rates),
            "shortage_refill": {
                "triggered_ratio": _ratio(shortage_refill_triggered, total),
                "added": _summarize(shortage_refill_added_values),
            },
            "top_k_raw_source": first_meta.get("top_k_raw_source"),
            "overfetch_factor": overfetch_factor,
            "dedup_key_strategy": "chunk_id > doc_id+sentence_idx > title+sentence_idx > title+text_hash",
            "chunk_fallback": {
                "top_k": first_meta.get("chunk_fallback_top_k"),
                "source": first_meta.get("chunk_fallback_top_k_source"),
            },
        },
        "official": {
            "pred_path": str(pred_official_out),
            "pred_topk_path": str(pred_official_topk_out) if has_pred_sp_topk else "",
            "gold_path": str(gold_official_out),
        },
        "audit": {
            "snapshot_path": str(snapshot_path),
            "report_path": str(output_dir / "alignment_audit_report.md"),
        },
    }

    _write_json(output_dir / "alignment_manifest.json", manifest)

    def _status(label: str, ratio: float) -> str:
        if label in {"pred_topk_eq_topk_context", "pred_subset_topk"}:
            return "OK" if ratio == 1.0 else "RED"
        if label == "pred_eq_topk_context":
            if str(pred_sp_policy).strip().lower() != "topk":
                return "INFO"
            return "OK" if ratio == 1.0 else "RED"
        if ratio == 1.0:
            return "OK"
        return "YELLOW"

    report_lines = [
        "# Hotpot Alignment Audit Report",
        "",
        "## Field Flow",
        "- gold_sp: data.supporting_facts -> record.gold_sp (supporting_facts alias)",
        "- pred_sp: policy-selected supporting facts -> record.pred_sp (sp alias)",
        "- pred_sp_topk: legacy retrieved_context_topk mapping for compatibility",
        "- official_pred.json: answer=short_answer, sp=pred_sp",
        "- official_pred_topk.json: answer=short_answer, sp=pred_sp_topk (if available)",
        "- official_gold.json: _id/answer/supporting_facts from gold jsonl",
        "",
        "## Sanity Checks (D1 closed-context)",
        f"- P(pred_sp == gold_sp): {pred_eq_gold_ratio:.4f} ({pred_eq_gold}/{total}) [{_status('pred_eq_gold', pred_eq_gold_ratio)}]",
        f"- P(pred_sp == retrieved_context_topk): {pred_eq_topk_context_ratio:.4f} ({pred_eq_topk_context}/{total}) [{_status('pred_eq_topk_context', pred_eq_topk_context_ratio)}]",
        f"- P(pred_sp_topk == retrieved_context_topk): {pred_topk_eq_topk_context_ratio:.4f} ({pred_topk_eq_topk_context}/{total}) [{_status('pred_topk_eq_topk_context', pred_topk_eq_topk_context_ratio)}]",
        f"- P(pred_sp subset pred_sp_topk): {pred_subset_topk_ratio:.4f} ({pred_subset_topk}/{total}) [{_status('pred_subset_topk', pred_subset_topk_ratio)}]",
        f"- P(gold_sp subset retrieved_context_topk): {gold_subset_ratio:.4f} ({gold_subset_topk}/{total}) [{_status('gold_subset', gold_subset_ratio)}]",
        f"- pred_sp_policy: {pred_sp_policy} (mixed={pred_sp_policy_is_mixed})",
        "- Note: closed-context subset matches are expected; this is not leakage.",
        "",
        "## Answer Source Distribution",
        "| source | count | ratio |",
        "| --- | --- | --- |",
    ]
    for source, count in sorted(answer_sources.items(), key=lambda item: (-item[1], item[0])):
        report_lines.append(f"| {source} | {count} | {_ratio(count, total):.4f} |")

    report_lines.extend(
        [
            "",
            "## Top-K / Dedup Stats",
            f"- top_k (mean/min/max): {_summarize(top_k_values)}",
            f"- top_k_raw (mean/min/max): {_summarize(top_k_raw_values)}",
            f"- top_k_final (mean/min/max): {_summarize(top_k_final_values)}",
            f"- duplicate_rate (mean/min/max): {_summarize(duplicate_rates)}",
            f"- shortage_refill.triggered_ratio: {_ratio(shortage_refill_triggered, total):.4f}",
            f"- shortage_refill.added (mean/min/max): {_summarize(shortage_refill_added_values)}",
            "",
            "## Scope",
            "- D1 closed-context only; D2 open-domain not executed.",
        ]
    )

    report_path = output_dir / "alignment_audit_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[alignment] wrote {pred_official_out}")
    if has_pred_sp_topk:
        print(f"[alignment] wrote {pred_official_topk_out}")
    print(f"[alignment] wrote {gold_official_out}")
    print(f"[alignment] wrote {snapshot_path}")
    print(f"[alignment] wrote {output_dir / 'alignment_manifest.json'}")
    print(f"[alignment] wrote {report_path}")


if __name__ == "__main__":
    main()
