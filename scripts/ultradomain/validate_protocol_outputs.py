from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from scripts.ultradomain.common import OUTPUT_ROOT, read_json, read_jsonl, ultradomain_get, write_json

SYSTEMS = ["RelRAG-full", "BM25-only", "Dense-only", "Hybrid-only"]
PAIRS = [
    ("RelRAG-full", "BM25-only"),
    ("RelRAG-full", "Dense-only"),
    ("RelRAG-full", "Hybrid-only"),
]
DIMENSIONS = ["comprehensiveness", "diversity", "empowerment", "overall"]
ALLOWED_WINNERS = {"A", "B", "tie"}


def _read_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    questions = payload.get("questions") if isinstance(payload, dict) else None
    return questions if isinstance(questions, list) else []


def _qid_set(records: List[Dict[str, Any]]) -> Set[str]:
    qids: Set[str] = set()
    for row in records:
        qid = row.get("question_id")
        if isinstance(qid, str) and qid:
            qids.add(qid)
    return qids


def _winner_valid(row: Dict[str, Any]) -> bool:
    winner = row.get("winner")
    if not isinstance(winner, dict):
        return False
    for dim in DIMENSIONS:
        val = winner.get(dim)
        if val not in ALLOWED_WINNERS:
            return False
    return True


def _check_answer_row_meta(row: Dict[str, Any]) -> bool:
    meta = row.get("run_meta")
    if not isinstance(meta, dict):
        return False
    required = ["prompt_template_hash", "prompt_instance_hash", "config_snapshot"]
    return all(key in meta for key in required)


def _check_judge_row_meta(row: Dict[str, Any]) -> bool:
    meta = row.get("run_meta")
    if not isinstance(meta, dict):
        return False
    required = ["config_snapshot", "ab_bias_control", "tie_policy"]
    if not all(key in meta for key in required):
        return False
    return "prompt_template_hash" in row and "prompt_instance_hash" in row


def _record_missing(path: Path, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def main() -> None:
    expected_q_default = int(ultradomain_get("validation.expected_questions_per_domain", 125) or 125)
    strict_default = bool(ultradomain_get("validation.strict", True))
    protocol_version_default = ultradomain_get("protocol.version", "v1")

    parser = argparse.ArgumentParser(description="Validate UltraDomain protocol outputs.")
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT))
    parser.add_argument("--protocol_version", default=protocol_version_default)
    parser.add_argument("--expected_questions_per_domain", type=int, default=expected_q_default)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=strict_default,
        help="Exit non-zero on validation errors.",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    run_meta = root / "run_meta"
    chunks_dir = root / "chunks"
    questions_dir = root / "questions"
    answers_dir = root / "answers"
    judge_dir = root / "judge"
    summary_dir = root / "summary"

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "output_root": str(root),
        "domains": {},
    }

    # Core required files.
    _record_missing(run_meta / "system_configs.json", errors)
    _record_missing(run_meta / "run_config.json", errors)
    _record_missing(run_meta / "protocol_version", errors)
    _record_missing(chunks_dir / "chunk_stats.json", errors)
    _record_missing(questions_dir / "questions_manifest.json", errors)
    _record_missing(summary_dir / "summary_winrate.md", errors)

    proto_path = run_meta / "protocol_version"
    if proto_path.exists():
        version = proto_path.read_text(encoding="utf-8").strip()
        if version != args.protocol_version:
            errors.append(
                f"protocol version mismatch: expected={args.protocol_version} actual={version}"
            )

    for domain in ("mix", "legal"):
        dom_stats: Dict[str, Any] = {}
        stats["domains"][domain] = dom_stats

        chunk_path = chunks_dir / f"chunks_{domain}.jsonl"
        _record_missing(chunk_path, errors)
        chunk_rows = _load_jsonl(chunk_path)
        dom_stats["chunk_count"] = len(chunk_rows)
        if chunk_rows and not any((row.get("text") or "").strip() for row in chunk_rows):
            warnings.append(f"{domain}: chunk file exists but all chunk text are empty")

        question_path = questions_dir / f"questions_{domain}.json"
        _record_missing(question_path, errors)
        questions = _read_questions(question_path)
        dom_stats["question_count"] = len(questions)
        if len(questions) != args.expected_questions_per_domain:
            errors.append(
                f"{domain}: expected {args.expected_questions_per_domain} questions, got {len(questions)}"
            )
        qids = [q.get("question_id") for q in questions if isinstance(q, dict)]
        qid_set = {qid for qid in qids if isinstance(qid, str) and qid}
        if len(qid_set) != len(questions):
            errors.append(f"{domain}: question_id is missing or duplicated")

        # Answers coverage + audit fields.
        answers_stats: Dict[str, Any] = {}
        dom_stats["answers"] = answers_stats
        for system in SYSTEMS:
            answer_path = answers_dir / domain / f"{system}.jsonl"
            _record_missing(answer_path, errors)
            rows = _load_jsonl(answer_path)
            row_qids = _qid_set(rows)
            missing = sorted(qid_set - row_qids)
            extra = sorted(row_qids - qid_set)
            bad_meta = sum(1 for row in rows if not _check_answer_row_meta(row))
            answers_stats[system] = {
                "rows": len(rows),
                "missing_questions": len(missing),
                "extra_questions": len(extra),
                "rows_missing_audit_fields": bad_meta,
            }
            if missing:
                errors.append(f"{domain}/{system}: missing answers for {len(missing)} questions")
            if extra:
                warnings.append(f"{domain}/{system}: has {len(extra)} extra answers not in questions file")
            if bad_meta:
                errors.append(
                    f"{domain}/{system}: {bad_meta} rows missing prompt hash/config snapshot fields"
                )
            meta_path = answers_dir / domain / f"{system}.meta.json"
            _record_missing(meta_path, errors)

        # Judge coverage + output validity + audit fields.
        judge_stats: Dict[str, Any] = {}
        dom_stats["judge"] = judge_stats
        for left, right in PAIRS:
            pair_name = f"{left}_vs_{right}"
            judge_path = judge_dir / domain / f"{pair_name}.jsonl"
            _record_missing(judge_path, errors)
            rows = _load_jsonl(judge_path)
            row_qids = _qid_set(rows)
            missing = sorted(qid_set - row_qids)
            invalid_winner = sum(1 for row in rows if not _winner_valid(row))
            bad_meta = sum(1 for row in rows if not _check_judge_row_meta(row))
            judge_stats[pair_name] = {
                "rows": len(rows),
                "missing_questions": len(missing),
                "invalid_winner_rows": invalid_winner,
                "rows_missing_audit_fields": bad_meta,
            }
            if missing:
                errors.append(f"{domain}/{pair_name}: missing judge rows for {len(missing)} questions")
            if invalid_winner:
                errors.append(f"{domain}/{pair_name}: {invalid_winner} rows have invalid winner schema")
            if bad_meta:
                errors.append(
                    f"{domain}/{pair_name}: {bad_meta} rows missing prompt hash/config snapshot fields"
                )
            meta_path = judge_dir / domain / f"{pair_name}.meta.json"
            _record_missing(meta_path, errors)

    report = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    out_path = run_meta / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, report)

    print(f"Validation report written to: {out_path}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    if errors:
        for msg in errors:
            print(f"[ERROR] {msg}")
    if warnings:
        for msg in warnings:
            print(f"[WARN] {msg}")

    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
