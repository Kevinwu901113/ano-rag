from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval_retrieval_v3.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_v3_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_text_only_mode_outputs_null_ie_ndcg(tmp_path: Path) -> None:
    mod = _load_eval_module()
    gold_file = tmp_path / "gold.jsonl"
    pred_file = tmp_path / "pred.jsonl"

    _write_jsonl(
        gold_file,
        [
            {"id": "q1", "supporting_facts": [["Alpha", 0], ["Beta", 1]]},
            {"id": "q2", "supporting_facts": [["Gamma", 0]]},
        ],
    )
    _write_jsonl(
        pred_file,
        [
            {"id": "q1", "ctxs": ["Alpha is mentioned.", "Beta is also mentioned."]},
            {"id": "q2", "ctxs": ["This context has no target title."]},
        ],
    )

    result = mod.evaluate_retrieval(
        pred_file=pred_file,
        gold_file=gold_file,
        expected_count=2,
    )

    assert result["alignment_ok"] is True
    assert result["recall_mode"] == "text"
    assert result["recall_text"] is not None
    assert result["recall@2"] == result["recall@5"]
    assert result["ie@2"] is None
    assert result["ie@5"] is None
    assert result["ndcg@2"] is None
    assert result["ndcg@5"] is None


def test_structured_ranked_mode_has_rank_metrics(tmp_path: Path) -> None:
    mod = _load_eval_module()
    gold_file = tmp_path / "gold.jsonl"
    pred_file = tmp_path / "pred.jsonl"

    _write_jsonl(
        gold_file,
        [
            {"id": "q1", "supporting_facts": [["Alpha", 0], ["Beta", 1]]},
            {"id": "q2", "supporting_facts": [["Gamma", 0]]},
        ],
    )
    _write_jsonl(
        pred_file,
        [
            {
                "id": "q1",
                "ctxs": [
                    {"id": "d1", "title": "Alpha", "text": "A", "rank": 1},
                    {"id": "d2", "title": "Noise", "text": "N", "rank": 2},
                    {"id": "d3", "title": "Beta", "text": "B", "rank": 3},
                ],
            },
            {
                "id": "q2",
                "ctxs": [
                    {"id": "d4", "title": "Noise", "text": "N", "rank": 1},
                    {"id": "d5", "title": "Gamma", "text": "G", "rank": 2},
                ],
            },
        ],
    )

    result = mod.evaluate_retrieval(
        pred_file=pred_file,
        gold_file=gold_file,
        expected_count=2,
    )

    assert result["alignment_ok"] is True
    assert result["recall_mode"] == "structured_ranked"
    assert result["recall@5"] >= result["recall@2"]
    assert result["ie@2"] is not None
    assert result["ie@5"] is not None
    assert result["ndcg@2"] is not None
    assert result["ndcg@5"] is not None


def test_alignment_mismatch_marks_group_na(tmp_path: Path) -> None:
    mod = _load_eval_module()
    gold_file = tmp_path / "gold.jsonl"
    pred_file = tmp_path / "pred.jsonl"

    _write_jsonl(
        gold_file,
        [
            {"id": "q1", "supporting_facts": [["Alpha", 0]]},
            {"id": "q2", "supporting_facts": [["Beta", 0]]},
            {"id": "q3", "supporting_facts": [["Gamma", 0]]},
        ],
    )
    _write_jsonl(
        pred_file,
        [
            {"id": "q1", "ctxs": [{"id": "d1", "title": "Alpha", "text": "A", "rank": 1}]},
            {"id": "q2", "ctxs": [{"id": "d2", "title": "Beta", "text": "B", "rank": 1}]},
        ],
    )

    result = mod.evaluate_retrieval(
        pred_file=pred_file,
        gold_file=gold_file,
        expected_count=3,
    )

    assert result["id_overlap_count"] == 2
    assert result["evaluated_count"] == 2
    assert result["alignment_ok"] is False
    assert result["recall@2"] is None
    assert result["recall@5"] is None
    assert result["ie@2"] is None
    assert result["ie@5"] is None
    assert result["ndcg@2"] is None
    assert result["ndcg@5"] is None
    assert result["recall_text"] is None
