import json
from pathlib import Path

from relrag.indexer.index_builder import IndexBuilder
from relrag.retriever.pipeline import retrieve_answer
from relrag.retriever.operators import Indexes
from relrag.retriever.note_store import NoteStore


def _write_notes_and_indexes(tmp_path: Path) -> tuple[Path, Path]:
    notes_path = tmp_path / "notes.jsonl"
    dummy_note = {
        "note_id": "dummy#c0000#0",
        "subj": "Dummy",
        "pred": "occupation",
        "obj": "placeholder",
        "subj_type": "PERSON",
        "obj_type": "CONCEPT",
        "evidence": "Dummy is a placeholder.",
        "meta": {
            "source": "dummy#c0000",
            "confidence": 0.9,
            "subject_profile": {"type": "PERSON", "aliases": []},
            "attribute": {"name": "occupation", "values": [{"value": "placeholder"}]},
        },
    }
    with notes_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dummy_note, ensure_ascii=False) + "\n")
    index_dir = tmp_path / "indexes"
    builder = IndexBuilder()
    builder.build_from_jsonl(str(notes_path))
    builder.dump(str(index_dir))
    return notes_path, index_dir


def _write_chunks(tmp_path: Path, chunks: list[dict]) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _cfg_no_hybrid() -> dict:
    return {
        "retriever": {
            "structured": {"enabled": True, "vector_fallback_enabled": False},
            "hybrid": {"enabled": False},
            "embedding": {"enabled": False},
            "bm25": {"enabled": False},
            "chunk_fallback": {"top_k": 6},
        }
    }


def test_compare_species_chunk_fallback(tmp_path: Path) -> None:
    notes_path, index_dir = _write_notes_and_indexes(tmp_path)
    _write_chunks(
        tmp_path,
        [
            {
                "doc_id": "greyia_doc",
                "chunk_id": "c0000",
                "text": "Greyia contains three species.",
                "meta": {"sent_spans": [{"text": "Greyia contains three species."}]},
            },
            {
                "doc_id": "calibanus_doc",
                "chunk_id": "c0001",
                "text": "Calibanus is a genus of two species.",
                "meta": {"sent_spans": [{"text": "Calibanus is a genus of two species."}]},
            },
        ],
    )
    indexes = Indexes(str(index_dir))
    note_store = NoteStore(str(notes_path))
    result = retrieve_answer(
        "Between Greyia and Calibanus, which genus contains more species?",
        indexes,
        note_store,
        cfg=_cfg_no_hybrid(),
    )
    assert result.get("evidence")
    joined = " ".join(ev.get("evidence", "") for ev in result["evidence"])
    assert "species" in joined.lower()


def test_count_members_chunk_fallback(tmp_path: Path) -> None:
    notes_path, index_dir = _write_notes_and_indexes(tmp_path)
    _write_chunks(
        tmp_path,
        [
            {
                "doc_id": "florida_senate",
                "chunk_id": "c0000",
                "text": "The Florida Senate has 40 members.",
                "meta": {"sent_spans": [{"text": "The Florida Senate has 40 members."}]},
            }
        ],
    )
    indexes = Indexes(str(index_dir))
    note_store = NoteStore(str(notes_path))
    result = retrieve_answer(
        "How many members does the Florida Senate have?",
        indexes,
        note_store,
        cfg=_cfg_no_hybrid(),
    )
    assert result.get("evidence")
    joined = " ".join(ev.get("evidence", "") for ev in result["evidence"])
    assert "40" in joined


def test_professor_university_founded_chunk_fallback(tmp_path: Path) -> None:
    notes_path, index_dir = _write_notes_and_indexes(tmp_path)
    _write_chunks(
        tmp_path,
        [
            {
                "doc_id": "tokarev_doc",
                "chunk_id": "c0000",
                "text": "Vladimir Tokarev was a professor at Saint Petersburg State University.",
                "meta": {
                    "sent_spans": [
                        {"text": "Vladimir Tokarev was a professor at Saint Petersburg State University."}
                    ]
                },
            },
            {
                "doc_id": "spbsu_doc",
                "chunk_id": "c0001",
                "text": "Saint Petersburg State University was founded in 1724.",
                "meta": {"sent_spans": [{"text": "Saint Petersburg State University was founded in 1724."}]},
            },
        ],
    )
    indexes = Indexes(str(index_dir))
    note_store = NoteStore(str(notes_path))
    result = retrieve_answer(
        "The university where Vladimir Tokarev was a professor was founded in what year?",
        indexes,
        note_store,
        cfg=_cfg_no_hybrid(),
    )
    assert result.get("evidence")
    evidence_texts = [ev.get("evidence", "").lower() for ev in result["evidence"]]
    assert any("professor" in text for text in evidence_texts)
    assert any("founded" in text for text in evidence_texts)
