from __future__ import annotations

from pathlib import Path

import pytest

from answer.efsa_answer import efsa_answer_with_fallback
from utils.file_utils import FileUtils


@pytest.mark.parametrize(
    "candidates_in_memory",
    [
        [],
        [{"note_id": "legacy", "content": "should not be used"}],
    ],
)
def test_final_recall_file_overrides_candidates(monkeypatch, tmp_path, candidates_in_memory):
    final_recall_path = Path(tmp_path) / "final_recall.jsonl"
    file_candidates = [
        {"note_id": "n1", "content": "first"},
        {"note_id": "n2", "content": "second"},
    ]
    FileUtils.write_jsonl(file_candidates, str(final_recall_path))

    observed_note_ids: list[str] = []

    def fake_efsa(candidates, query, bridge_entity=None, path_entities=None, topN=20):  # noqa: D401
        observed_note_ids.extend(note.get("note_id") for note in candidates)
        return "answer", []

    monkeypatch.setattr("answer.efsa_answer.efsa_answer", fake_efsa)

    answer, supports = efsa_answer_with_fallback(
        candidates=candidates_in_memory,
        query="test",
        final_recall_path=str(final_recall_path),
    )

    assert answer == "answer"
    assert supports == []
    assert observed_note_ids == ["n1", "n2"], "EFSA should consume candidates from final_recall.jsonl"
