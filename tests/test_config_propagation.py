import importlib
import sys
from pathlib import Path

import numpy as np
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "GPUtil" not in sys.modules:
    sys.modules["GPUtil"] = types.SimpleNamespace(getGPUs=lambda: [])

from config.config_loader import ConfigLoader


def _patch_query_processor(monkeypatch, loader):
    import query.query_processor as qp

    monkeypatch.setattr("config.config_loader.config", loader, raising=False)
    monkeypatch.setattr(qp, "config", loader, raising=False)

    class DummyVectorRetriever:
        def __init__(self):
            self.atomic_notes = []
            self.note_embeddings = np.zeros((0, 1))

        def build_index(self, atomic_notes):
            self.atomic_notes = atomic_notes
            self.note_embeddings = np.zeros((len(atomic_notes), 1))
            return True

    class DummyGraphBuilder:
        def __init__(self, llm=None):
            self.llm = llm

        def build_graph(self, atomic_notes, embeddings):
            return {}

    class DummyGraphIndex:
        def build_index(self, graph, atomic_notes, embeddings):
            return True

    class DummyGraphRetriever:
        def __init__(self, *args, **kwargs):
            pass

    class DummyContextDispatcher:
        def __init__(self, config, **kwargs):
            self._config = config

        def dispatch(self, candidates, query=None):
            return candidates

    class DummyEnhancedRecallOptimizer:
        def __init__(self, *args, **kwargs):
            pass

    class DummyMultiHopProcessor:
        def __init__(self, *args, **kwargs):
            pass

    class DummyScheduler:
        def __init__(self, *args, **kwargs):
            pass

    class DummyRewriter:
        def __init__(self, *args, **kwargs):
            self.enabled = False

    class DummyDiversityScheduler:
        def __init__(self, config):
            self.config = config

    class DummyContextPacker:
        def __init__(self, *args, **kwargs):
            pass

    def _simple_factory(*args, **kwargs):
        return object()

    monkeypatch.setattr(qp, "VectorRetriever", DummyVectorRetriever, raising=False)
    monkeypatch.setattr(qp, "GraphBuilder", DummyGraphBuilder, raising=False)
    monkeypatch.setattr(qp, "GraphIndex", DummyGraphIndex, raising=False)
    monkeypatch.setattr(qp, "GraphRetriever", DummyGraphRetriever, raising=False)
    monkeypatch.setattr(qp, "ContextDispatcher", DummyContextDispatcher, raising=False)
    monkeypatch.setattr(qp, "EnhancedRecallOptimizer", DummyEnhancedRecallOptimizer, raising=False)
    monkeypatch.setattr(qp, "MultiHopQueryProcessor", DummyMultiHopProcessor, raising=False)
    monkeypatch.setattr(qp, "MultiHopContextScheduler", DummyScheduler, raising=False)
    monkeypatch.setattr(qp, "ContextScheduler", DummyScheduler, raising=False)
    monkeypatch.setattr(qp, "create_path_aware_ranker", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "create_learned_fusion", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "create_qa_coverage_scorer", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "create_span_picker", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "create_answer_verifier", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "ContextPacker", DummyContextPacker, raising=False)
    monkeypatch.setattr(qp, "create_listt5_reranker", _simple_factory, raising=False)
    monkeypatch.setattr(qp, "LLMBasedRewriter", DummyRewriter, raising=False)
    monkeypatch.setattr(qp, "DiversityScheduler", DummyDiversityScheduler, raising=False)
    monkeypatch.setattr(qp, "build_bm25_corpus", lambda notes, func: {i: func(n) for i, n in enumerate(notes)}, raising=False)
    # 移除对 OllamaClient 的 monkeypatch；不再需要
    # 保留其他组件的占位或模拟（如有）

    return qp


def _write_config(tmp_path: Path, hybrid_enabled: bool, dispatcher_enabled: bool) -> Path:
    path = tmp_path / f"config_{int(hybrid_enabled)}_{int(dispatcher_enabled)}.yaml"
    path.write_text("")  # ensure file exists before dumping defaults
    with path.open("w", encoding="utf-8") as handle:
        handle.write("hybrid_search:\n  enabled: {0}\ncontext_dispatcher:\n  enabled: {1}\n".format(
            str(hybrid_enabled).lower(), str(dispatcher_enabled).lower()
        ))
    return path


def test_config_toggle_propagates_into_processor(monkeypatch, tmp_path):
    import query.query_processor as qp

    importlib.reload(qp)

    disabled_path = _write_config(tmp_path, hybrid_enabled=False, dispatcher_enabled=False)
    loader_disabled = ConfigLoader(str(disabled_path))

    qp_disabled = _patch_query_processor(monkeypatch, loader_disabled)
    notes = [{"note_id": "1", "title": "A", "raw_span": "B", "content": "C"}]
    processor_disabled = qp_disabled.QueryProcessor(notes)

    assert loader_disabled.get("hybrid_search.enabled") is False
    assert loader_disabled.get("retrieval.hybrid.enabled") is False
    assert loader_disabled.get("context_dispatcher.enabled") is False
    assert processor_disabled.hybrid_search_enabled is False
    assert processor_disabled.use_context_dispatcher is False

    enabled_path = _write_config(tmp_path, hybrid_enabled=True, dispatcher_enabled=True)
    loader_enabled = ConfigLoader(str(enabled_path))

    qp_enabled = _patch_query_processor(monkeypatch, loader_enabled)
    processor_enabled = qp_enabled.QueryProcessor(notes)

    assert loader_enabled.get("hybrid_search.enabled") is True
    assert loader_enabled.get("retrieval.hybrid.enabled") is True
    assert loader_enabled.get("context_dispatcher.enabled") is True
    assert processor_enabled.hybrid_search_enabled is True
    assert processor_enabled.use_context_dispatcher is True
