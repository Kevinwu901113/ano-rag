from collections import defaultdict
from pathlib import Path
import importlib.util


def _load_expand():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "relrag" / "retriever" / "operators.py"
    spec = importlib.util.spec_from_file_location("operators_mod", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod.EXPAND_from


class _DummyIndexes:
    def __init__(self) -> None:
        self.graph_edges = defaultdict(list)
        self.inverse_edges = defaultdict(list)


def test_expand_from_with_none_predicate_returns_edges():
    expand_from = _load_expand()
    idx = _DummyIndexes()
    idx.graph_edges["A"] = [
        {"pred": "p1", "obj": "B", "note_id": "n1", "conf": 0.8},
        {"pred": "p2", "obj": "C", "note_id": "n2", "conf": 0.7},
    ]
    out = expand_from(idx, "A", predicate=None, direction="out", limit=10)
    assert len(out) == 2
    assert out[0][0] == "B"
    assert out[1][0] == "C"


def test_expand_from_with_predicate_filters_normally():
    expand_from = _load_expand()
    idx = _DummyIndexes()
    idx.graph_edges["A"] = [
        {"pred": "p1", "obj": "B", "note_id": "n1", "conf": 0.8},
        {"pred": "p2", "obj": "C", "note_id": "n2", "conf": 0.7},
    ]
    out = expand_from(idx, "A", predicate="p2", direction="out", limit=10)
    assert len(out) == 1
    assert out[0][0] == "C"
