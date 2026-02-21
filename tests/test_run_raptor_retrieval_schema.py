from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_schema_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "baseline" / "runners" / "retrieval_schema.py"
    spec = importlib.util.spec_from_file_location("retrieval_schema_mod", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def test_raptor_ctxs_required_fields_and_min_topk() -> None:
    mod = _load_schema_module()
    docs = [
        {"id": "d1", "title": "Doc 1", "text": "Doc one body."},
        {"id": "d2", "title": "Doc 2", "text": "Doc two body."},
        {"id": "d3", "title": "Doc 3", "text": "Doc three body."},
    ]
    context_text = (
        "### DOC d1 | Title: Doc 1\nDoc one body.\n"
        "### DOC d2 | Title: Doc 2\nDoc two body.\n"
        "### DOC d3 | Title: Doc 3\nDoc three body.\n"
    )

    ctxs = mod.build_raptor_ctxs(
        context_text=context_text,
        docs=docs,
        top_k=5,
        layer_information=[],
        node_text_by_index={},
    )

    assert len(ctxs) == min(5, len(docs))
    for expected_rank, row in enumerate(ctxs, start=1):
        assert isinstance(row, dict)
        assert {"id", "title", "text", "rank"}.issubset(row.keys())
        assert row["rank"] == expected_rank
        assert isinstance(row["text"], str) and row["text"]


def test_raptor_ctxs_use_layer_info_provenance() -> None:
    mod = _load_schema_module()
    docs = [
        {"id": "d1", "title": "Doc 1", "text": "Doc one body."},
        {"id": "d2", "title": "Doc 2", "text": "Doc two body."},
        {"id": "d3", "title": "Doc 3", "text": "Doc three body."},
    ]
    layer_information = [
        {"node_index": 11, "layer_number": 2},
        {"node_index": 12, "layer_number": 2},
        {"node_index": 13, "layer_number": 2},
    ]
    node_text_by_index = {
        11: "### DOC d1 | Title: Doc 1\nDoc one body.\n",
        12: "### DOC d2 | Title: Doc 2\nDoc two body.\n",
        13: "### DOC d3 | Title: Doc 3\nDoc three body.\n",
    }

    ctxs = mod.build_raptor_ctxs(
        context_text="",
        docs=docs,
        top_k=2,
        layer_information=layer_information,
        node_text_by_index=node_text_by_index,
    )

    assert len(ctxs) == 2
    for row in ctxs:
        assert {"id", "title", "text", "rank"}.issubset(row.keys())
        provenance = row.get("provenance") or {}
        assert "node_index" in provenance
        assert "layer_number" in provenance
