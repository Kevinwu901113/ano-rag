import json

import pytest

pytest.importorskip("filelock")

from hotpot_entry import _build_pred_sp, _normalize_pred_sp_policy, _write_chunks_for_example


def test_pred_sp_policy_normalization():
    assert _normalize_pred_sp_policy("topk") == "topk"
    assert _normalize_pred_sp_policy("legacy_topk") == "topk"
    assert _normalize_pred_sp_policy("high-confidence") == "high_confidence"


def test_build_pred_sp_high_confidence_filters_and_caps():
    retrieved_context = [
        {"title": "Alpha", "sentence_idx": 0, "score": 0.9, "weak": False},
        {"title": "Alpha", "sentence_idx": 1, "score": 0.8, "weak": False},
        {"title": "Beta", "sentence_idx": 2, "score": 0.85, "weak": True},
        {"title": "Gamma", "sentence_idx": 3, "score": 0.7, "weak": False},
    ]
    pred_sp, pred_sp_topk, meta = _build_pred_sp(
        retrieved_context,
        policy="high_confidence",
        max_facts=2,
        min_score=0.0,
        drop_weak=True,
        prefer_new_titles=True,
    )
    assert pred_sp_topk == [["Alpha", 0], ["Alpha", 1], ["Beta", 2], ["Gamma", 3]]
    assert pred_sp == [["Alpha", 0], ["Gamma", 3]]
    assert meta["policy"] == "high_confidence"
    assert meta["dropped_weak"] == 1
    assert meta["selected_count"] == 2


def test_build_pred_sp_topk_compatibility():
    retrieved_context = [
        {"title": "DocA", "sentence_idx": 4, "score": 0.2},
        {"title": "DocB", "sentence_idx": 1, "score": 0.1},
    ]
    pred_sp, pred_sp_topk, meta = _build_pred_sp(
        retrieved_context,
        policy="topk",
        max_facts=0,
        min_score=0.0,
        drop_weak=False,
        prefer_new_titles=False,
    )
    assert pred_sp == [["DocA", 4], ["DocB", 1]]
    assert pred_sp_topk == pred_sp
    assert meta["policy"] == "topk"


def test_write_chunks_for_example_exports_sent_spans(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    doc_index = {
        "q1_00_doc": {
            "title": "Doc",
            "sentences": ["Sentence one.", "Sentence two."],
        }
    }
    written = _write_chunks_for_example(doc_index, chunks_path, overwrite=True)
    assert written == 1
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == "q1_00_doc"
    assert row["meta"]["sent_spans"][1]["text"] == "Sentence two."
