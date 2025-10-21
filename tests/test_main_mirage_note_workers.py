import sys
from pathlib import Path
import types

mirage_module = types.ModuleType("MIRAGE")
mirage_utils_module = types.ModuleType("MIRAGE.utils")


def _dummy_load_json(path):
    return []


def _dummy_convert_doc_pool(*args, **kwargs):
    return []


mirage_utils_module.load_json = _dummy_load_json
mirage_utils_module.convert_doc_pool = _dummy_convert_doc_pool
mirage_module.utils = mirage_utils_module

sys.modules.setdefault("MIRAGE", mirage_module)
sys.modules.setdefault("MIRAGE.utils", mirage_utils_module)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_mirage import MirageConfig, MirageRunner


def _make_config(tmp_path, max_workers_note):
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
        note_engines=["parallel"],
        enable_notes=True,
        enable_graph=False,
        max_workers_query=1,
        max_workers_note=max_workers_note,
        start_time=0.0,
    )


def test_parallel_generator_receives_max_workers(monkeypatch, tmp_path):
    recorded = {}

    class StubGenerator:
        def __init__(self, llm, max_workers=None):
            recorded["max_workers"] = max_workers

        def generate_atomic_notes(self, text_chunks):
            return [{"notes": [{"content": f"note-{i}"}]} for i, _ in enumerate(text_chunks)]

    monkeypatch.setattr("main_mirage.ParallelTaskAtomicNoteGenerator", StubGenerator)
    monkeypatch.setattr("main_mirage.LocalLLM", lambda: object())

    config = _make_config(tmp_path, max_workers_note=7)
    runner = MirageRunner(config)
    runner.doc_pool = [
        {"doc_chunk": "text", "doc_name": "doc", "mapped_id": "doc-1", "support": False}
    ]

    assert runner.generate_atomic_notes()
    assert recorded["max_workers"] == 7
    assert runner.doc_pool[0]["notes"][0]["content"] == "note-0"


def test_enhanced_generator_receives_max_workers_when_parallel_fails(monkeypatch, tmp_path):
    recorded = {}

    class FailingGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("parallel unavailable")

    class StubEnhancedGenerator:
        def __init__(self, llm, max_workers=None):
            recorded["max_workers"] = max_workers

        def generate_atomic_notes(self, text_chunks):
            return [{"notes": [{"content": "enhanced"}]} for _ in text_chunks]

    monkeypatch.setattr("main_mirage.ParallelTaskAtomicNoteGenerator", FailingGenerator)
    monkeypatch.setattr("main_mirage.EnhancedAtomicNoteGenerator", StubEnhancedGenerator)
    monkeypatch.setattr("main_mirage.LocalLLM", lambda: object())

    config = _make_config(tmp_path, max_workers_note=3)
    runner = MirageRunner(config)
    runner.doc_pool = [
        {"doc_chunk": "text", "doc_name": "doc", "mapped_id": "doc-1", "support": False}
    ]

    assert runner.generate_atomic_notes()
    assert recorded["max_workers"] == 3
    assert runner.doc_pool[0]["notes"][0]["content"] == "enhanced"


def test_basic_generator_receives_max_workers_when_all_fallbacks_fail(monkeypatch, tmp_path):
    recorded = {}

    class FailingGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("not available")

    class StubBasicGenerator:
        def __init__(self, llm, max_workers=None):
            recorded["max_workers"] = max_workers

        def generate_atomic_notes(self, text_chunks):
            return [{"notes": [{"content": "basic"}]} for _ in text_chunks]

    monkeypatch.setattr("main_mirage.ParallelTaskAtomicNoteGenerator", FailingGenerator)
    monkeypatch.setattr("main_mirage.EnhancedAtomicNoteGenerator", FailingGenerator)
    monkeypatch.setattr("main_mirage.AtomicNoteGenerator", StubBasicGenerator)
    monkeypatch.setattr("main_mirage.LocalLLM", lambda: object())

    config = _make_config(tmp_path, max_workers_note=5)
    runner = MirageRunner(config)
    runner.doc_pool = [
        {"doc_chunk": "text", "doc_name": "doc", "mapped_id": "doc-1", "support": False}
    ]

    assert runner.generate_atomic_notes()
    assert recorded["max_workers"] == 5
    assert runner.doc_pool[0]["notes"][0]["content"] == "basic"
