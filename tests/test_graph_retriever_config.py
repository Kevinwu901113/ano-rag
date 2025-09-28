import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import pytest

from config.config_loader import ConfigLoader
from graph.graph_index import GraphIndex


@pytest.fixture
def simple_graph_index():
    graph = nx.Graph()
    nodes = ["A", "B", "C", "D"]
    for node in nodes:
        graph.add_node(node, title=node, importance_score=1.0)
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=1.0)
    graph.add_edge("C", "D", weight=1.0)

    atomic_notes = [
        {"note_id": node, "title": node, "raw_span": node, "content": node}
        for node in nodes
    ]

    graph_index = GraphIndex(graph)
    graph_index.build_index(graph, atomic_notes)
    return graph_index


def _write_multi_hop_config(tmp_path: Path, max_hops: int) -> ConfigLoader:
    cfg_path = tmp_path / f"multi_hop_{max_hops}.yaml"
    cfg_path.write_text(
        "retrieval:\n  multi_hop:\n    enabled: true\n    max_hops: {0}\n".format(max_hops),
        encoding="utf-8",
    )
    return ConfigLoader(str(cfg_path))


def test_retrieval_multi_hop_config_controls_max_hops(monkeypatch, tmp_path, simple_graph_index):
    import graph.graph_retriever as gr

    importlib.reload(gr)

    short_loader = _write_multi_hop_config(tmp_path, max_hops=1)
    monkeypatch.setattr("config.config_loader.config", short_loader, raising=False)
    monkeypatch.setattr(gr, "config", short_loader, raising=False)

    short_retriever = gr.GraphRetriever(simple_graph_index)
    short_paths = short_retriever._bfs_reasoning_paths("A")
    assert short_retriever.max_hops == 1
    assert all(len(path) <= 2 for path in short_paths)

    long_loader = _write_multi_hop_config(tmp_path, max_hops=3)
    monkeypatch.setattr("config.config_loader.config", long_loader, raising=False)
    monkeypatch.setattr(gr, "config", long_loader, raising=False)

    long_retriever = gr.GraphRetriever(simple_graph_index)
    long_paths = long_retriever._bfs_reasoning_paths("A")
    assert long_retriever.max_hops == 3
    assert any(len(path) > 2 for path in long_paths)


def test_legacy_multi_hop_config_is_used(monkeypatch, tmp_path, simple_graph_index):
    import graph.graph_retriever as gr

    importlib.reload(gr)

    cfg_path = tmp_path / "legacy_multi_hop.yaml"
    cfg_path.write_text(
        "multi_hop:\n  max_hops: 2\n",
        encoding="utf-8",
    )
    loader = ConfigLoader(str(cfg_path))

    monkeypatch.setattr("config.config_loader.config", loader, raising=False)
    monkeypatch.setattr(gr, "config", loader, raising=False)

    retriever = gr.GraphRetriever(simple_graph_index)
    assert retriever.max_hops == 2
    paths = retriever._bfs_reasoning_paths("A")
    assert any(len(path) == 3 for path in paths)
