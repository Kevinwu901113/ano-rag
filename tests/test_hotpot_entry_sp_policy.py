import json

import pytest

pytest.importorskip("filelock")

from hotpot_entry import (
    _build_pred_sp,
    _dedup_retrieved_context,
    _promote_second_title_by_question,
    _normalize_pred_sp_policy,
    _promote_title_diversity,
    _write_chunks_for_example,
)
from hotpot_entry import _iter_final_ranked_rows, normalize_title


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


def test_iter_final_ranked_rows_includes_required_fields():
    record = {
        "_id": "q1",
        "retrieved_context_topk": [
            {
                "doc_title": "  New   York City ",
                "chunk_id": "c1",
                "score": 0.5,
                "source": "hybrid",
                "text_hash": "abc",
                "note_id": "n1",
                "doc_id": "d1",
            }
        ],
    }
    rows = list(_iter_final_ranked_rows(record, top_k_export=50))
    assert len(rows) == 1
    row = rows[0]
    assert row["qid"] == "q1"
    assert row["rank"] == 1
    assert row["doc_title"] == "New York City"
    assert row["chunk_id"] == "c1"
    assert row["score"] == 0.5
    assert row["source"] == "hybrid"
    assert row["text_hash"] == "abc"
    assert normalize_title("  A   B  ") == "A B"


def test_iter_final_ranked_rows_does_not_fallback_to_doc_id_for_title():
    record = {
        "_id": "q2",
        "retrieved_context_topk": [
            {
                "doc_id": "internal_doc_id_only",
                "chunk_id": "c9",
                "score": 0.2,
            }
        ],
    }
    rows = list(_iter_final_ranked_rows(record, top_k_export=10))
    assert len(rows) == 1
    assert rows[0]["doc_title"] == ""


def test_promote_title_diversity_reorders_prefix_and_keeps_rank1():
    rows = [
        {"title": "A", "sentence_idx": 0},
        {"title": "A", "sentence_idx": 1},
        {"title": "B", "sentence_idx": 0},
        {"title": "C", "sentence_idx": 0},
    ]
    reordered, applied = _promote_title_diversity(rows, top_n=3, keep_first=True)
    assert applied is True
    assert [row["title"] for row in reordered] == ["A", "B", "C", "A"]


def test_dedup_retrieved_context_records_title_diversity_stats():
    contexts = [
        {"title": "Alpha", "sentence_idx": 0, "chunk_id": "a#0"},
        {"title": "Alpha", "sentence_idx": 1, "chunk_id": "a#1"},
        {"title": "Beta", "sentence_idx": 0, "chunk_id": "b#0"},
        {"title": "Gamma", "sentence_idx": 0, "chunk_id": "g#0"},
    ]
    topk, stats = _dedup_retrieved_context(
        contexts,
        top_k=3,
        title_diversity_enabled=True,
        title_diversity_top_n=3,
        title_diversity_keep_first=True,
    )
    assert [row["title"] for row in topk] == ["Alpha", "Beta", "Gamma"]
    assert stats["title_diversity_enabled"] is True
    assert stats["title_diversity_applied"] is True
    assert stats["title_diversity_top_n"] == 3


def test_promote_second_title_by_question_moves_matching_title():
    rows = [
        {"title": "Christopher Nolan", "sentence_idx": 0},
        {"title": "Tenet", "sentence_idx": 0},
        {"title": "Inception", "sentence_idx": 0},
    ]
    reordered, applied = _promote_second_title_by_question(
        rows,
        question="Which film is directed by Christopher Nolan, Inception or Tenet?",
        window_n=5,
    )
    assert applied is True
    assert [row["title"] for row in reordered] == ["Christopher Nolan", "Inception", "Tenet"]


def test_dedup_retrieved_context_records_query_title_promotion_stats():
    contexts = [
        {"title": "Christopher Nolan", "sentence_idx": 0, "chunk_id": "nolan#0"},
        {"title": "Tenet", "sentence_idx": 0, "chunk_id": "tenet#0"},
        {"title": "Inception", "sentence_idx": 0, "chunk_id": "inception#0"},
    ]
    topk, stats = _dedup_retrieved_context(
        contexts,
        top_k=3,
        title_diversity_enabled=False,
        question="Which film is directed by Christopher Nolan, Inception or Tenet?",
        query_title_promotion_enabled=True,
        query_title_promotion_window=5,
    )
    assert [row["title"] for row in topk] == ["Christopher Nolan", "Inception", "Tenet"]
    assert stats["query_title_promotion_enabled"] is True
    assert stats["query_title_promotion_applied"] is True
    assert stats["query_title_promotion_window"] == 5
