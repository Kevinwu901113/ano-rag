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


def test_graphrag_ctxs_are_built_from_sources() -> None:
    mod = _load_schema_module()
    ctxs = mod.build_graphrag_ctxs_from_sources(
        sources_records=[{"id": "tu-1", "text": "Source selected text"}],
        text_unit_records=[
            {
                "id": "tu-1",
                "human_readable_id": "tu-1",
                "document_id": "doc-1",
                "text": "Text-unit body",
            }
        ],
        document_records=[
            {
                "id": "doc-1",
                "title": "Doc A",
                "text": "Document body",
            }
        ],
        top_k=5,
    )

    assert isinstance(ctxs, list)
    assert len(ctxs) == 1
    assert not any(isinstance(item, str) for item in ctxs)

    row = ctxs[0]
    assert {"id", "title", "text", "rank"}.issubset(row.keys())
    assert row["title"] == "Doc A"
    assert row["text"] == "Source selected text"
    assert row["rank"] == 1
    assert row.get("provenance", {}).get("text_unit_id") == "tu-1"
