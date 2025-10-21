import json
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "MIRAGE.utils" not in sys.modules:
    import types

    mirage_pkg = types.ModuleType("MIRAGE")
    mirage_utils = types.ModuleType("MIRAGE.utils")

    def _load_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _convert_doc_pool(doc_pool):
        return doc_pool

    mirage_utils.load_json = _load_json
    mirage_utils.convert_doc_pool = _convert_doc_pool

    sys.modules["MIRAGE"] = mirage_pkg
    sys.modules["MIRAGE.utils"] = mirage_utils

from main_mirage import MirageConfig, MirageRunner


def create_basic_files(tmp_path: Path, include_oracle: bool):
    dataset_path = tmp_path / "dataset.json"
    doc_pool_path = tmp_path / "doc_pool.json"
    oracle_path = tmp_path / "oracle.json"

    dataset = [
        {"query_id": "q1", "query": "What is the capital?"},
        {"query_id": "q2", "query": "Who wrote the book?"},
    ]
    doc_pool = [
        {"mapped_id": "d1", "doc_name": "Doc 1", "doc_chunk": "Sample text."}
    ]
    oracle = {"q1": {"text": "Answer"}}

    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    doc_pool_path.write_text(json.dumps(doc_pool, indent=2), encoding="utf-8")
    oracle_payload = oracle if include_oracle else {}
    oracle_path.write_text(json.dumps(oracle_payload, indent=2), encoding="utf-8")

    return dataset_path, doc_pool_path, oracle_path


def build_runner(tmp_path: Path, mode: str) -> MirageRunner:
    dataset_path, doc_pool_path, oracle_path = create_basic_files(tmp_path, include_oracle=(mode == "oracle"))

    config = MirageConfig(
        run_id="test-run",
        mode=mode,
        topk=1,
        new_run=True,
        debug=False,
        dataset_path=str(dataset_path),
        doc_pool_path=str(doc_pool_path),
        oracle_path=str(oracle_path),
        result_dir=str(tmp_path / "results"),
        retriever_type="bm25",
        embed_model="",
        rebuild_index=False,
        model_name="dummy-model",
        temperature=0.0,
        max_tokens=16,
        seed=None,
        note_engines=[],
        enable_notes=False,
        enable_graph=False,
        max_workers_query=1,
        max_workers_note=1,
        start_time=time.time(),
    )

    runner = MirageRunner(config)
    runner.dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    runner.doc_pool = json.loads(doc_pool_path.read_text(encoding="utf-8"))
    runner.stats["total_queries"] = len(runner.dataset)

    if mode == "oracle":
        runner.oracle_data = json.loads(oracle_path.read_text(encoding="utf-8"))

    return runner


@pytest.mark.parametrize("mode", ["base", "oracle"])
def test_manifest_contains_file_stats(tmp_path, mode):
    runner = build_runner(tmp_path, mode)
    runner.save_manifest()

    manifest_path = runner.run_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json should be created"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_info = manifest.get("data_info", {})

    for field in [
        "dataset_file_size_bytes",
        "dataset_line_count",
        "doc_pool_file_size_bytes",
        "doc_pool_line_count",
    ]:
        assert field in data_info, f"Missing {field} in manifest data_info"
        assert isinstance(data_info[field], int), f"{field} should be an integer"
        assert data_info[field] > 0, f"{field} should be positive"

    logs_entry = manifest.get("output_files", {}).get("logs")
    assert logs_entry == ["logs/run.log", "logs/run_error.log"], "Log file names should match"

    if mode == "oracle":
        for field in [
            "oracle_file_size_bytes",
            "oracle_line_count",
        ]:
            assert field in data_info, f"Missing {field} for oracle mode"
            assert isinstance(data_info[field], int)
            assert data_info[field] > 0
    else:
        assert "oracle_file_size_bytes" not in data_info
        assert "oracle_line_count" not in data_info
