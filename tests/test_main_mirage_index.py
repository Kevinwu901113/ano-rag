import sys
import types
from pathlib import Path
from typing import List, Dict, Any

import logging
import pytest


def _ensure_mirage_stubs():
    """Provide lightweight MIRAGE modules required by main_mirage imports."""
    if "MIRAGE.utils" in sys.modules:
        return

    mirage_pkg = types.ModuleType("MIRAGE")
    utils_mod = types.ModuleType("MIRAGE.utils")

    def _load_json(path):  # pragma: no cover - stubbed dependency
        return []

    def _convert_doc_pool(doc_pool):  # pragma: no cover - stubbed dependency
        return doc_pool

    utils_mod.load_json = _load_json
    utils_mod.convert_doc_pool = _convert_doc_pool

    sys.modules["MIRAGE"] = mirage_pkg
    sys.modules["MIRAGE.utils"] = utils_mod


_ensure_mirage_stubs()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_mirage import MirageConfig, MirageRunner  # noqa: E402


def _make_config(tmp_path: Path, *, enable_notes: bool = True) -> MirageConfig:
    return MirageConfig(
        run_id="test-run",
        mode="mixed",
        topk=1,
        new_run=True,
        debug=False,
        dataset_path=str(tmp_path / "dataset.json"),
        doc_pool_path=str(tmp_path / "doc_pool.json"),
        oracle_path=str(tmp_path / "oracle.json"),
        result_dir=str(tmp_path / "results"),
        retriever_type="bm25",
        embed_model="",
        rebuild_index=False,
        model_name="dummy",
        temperature=0.0,
        max_tokens=128,
        seed=None,
        note_engines=[],
        enable_notes=enable_notes,
        enable_graph=False,
        max_workers_query=1,
        max_workers_note=1,
        start_time=0.0,
    )


def test_run_invokes_notes_before_index(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    runner = MirageRunner(config)

    order: List[str] = []

    def fake_load_data(self) -> bool:
        self.dataset = [{'query_id': 'q1', 'query': 'What?'}]
        self.doc_pool = [{'doc_chunk': 'chunk text', 'doc_name': 'Doc-1'}]
        return True

    def fake_generate(self) -> bool:
        order.append('notes')
        self.atomic_notes = [{
            'note_id': 'note-1',
            'chunk_id': 'Doc-1#0',
            'doc_name': 'Doc-1',
            'doc_hash': 'hash',
            'offsets': [0, 10],
            'content': 'note content',
            'metadata': {'note_id': 'note-1', 'chunk_id': 'Doc-1#0', 'doc_hash': 'hash', 'offsets': [0, 10]},
        }]
        return True

    def fake_build(self) -> bool:
        order.append('index')
        return True

    monkeypatch.setattr(MirageRunner, "load_data", fake_load_data, raising=False)
    monkeypatch.setattr(MirageRunner, "generate_atomic_notes", fake_generate, raising=False)
    monkeypatch.setattr(MirageRunner, "build_global_index", fake_build, raising=False)
    monkeypatch.setattr(MirageRunner, "build_graph", lambda self: True, raising=False)
    monkeypatch.setattr(MirageRunner, "initialize_llm", lambda self: True, raising=False)
    monkeypatch.setattr(MirageRunner, "run_parallel_processing", lambda self: [], raising=False)
    monkeypatch.setattr(MirageRunner, "save_predictions", lambda self, results: None, raising=False)
    monkeypatch.setattr(MirageRunner, "save_manifest", lambda self: None, raising=False)

    assert runner.run()
    assert order == ['notes', 'index']


def test_build_index_prefers_existing_notes(monkeypatch, tmp_path, caplog):
    config = _make_config(tmp_path)
    runner = MirageRunner(config)

    runner.doc_pool = [{
        'doc_chunk': 'Paragraph about topic.',
        'doc_name': 'Doc-1',
        'mapped_id': 'doc-1',
        'support': False,
        'notes': [{
            'note_id': 'cached-note',
            'content': 'LLM generated fact.',
            'metadata': {'chunk_id': 'Doc-1#0'}
        }]
    }]

    recorded: Dict[str, Any] = {}

    class DummyRetriever:
        def __init__(self, embedding_manager=None, retrieval_mode=None):
            self.embedding_manager = embedding_manager
            self.retrieval_mode = retrieval_mode
            self.data_dir = ''

        def set_embedding_model(self, model_name):
            return None

        def build_index(self, atomic_notes, force_rebuild=False, save_index=True):
            recorded['notes'] = atomic_notes
            return True

    monkeypatch.setattr("main_mirage.VectorRetriever", DummyRetriever)

    def fail_convert(self):  # pragma: no cover - should not be called
        raise AssertionError("convert_doc_pool_to_notes should not be used when notes exist")

    monkeypatch.setattr(MirageRunner, "convert_doc_pool_to_notes", fail_convert, raising=False)

    caplog.set_level(logging.INFO)

    assert runner.build_global_index()
    assert recorded['notes'], "atomic notes should be provided to retriever"
    assert recorded['notes'][0]['content'] == 'LLM generated fact.'
    assert any("atomic notes from doc_pool" in record.message for record in caplog.records)
