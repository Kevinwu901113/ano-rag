import unittest
import os, sys, types
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub heavy modules
sys.modules['loguru'] = types.SimpleNamespace(
    logger=types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
)
sys.modules['jsonlines'] = types.SimpleNamespace()
sys.modules['docx'] = types.SimpleNamespace(Document=lambda *a, **k: None)
sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **k: x)
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

saved_config = sys.modules.get('config')
sys.modules['config'] = types.SimpleNamespace(config=types.SimpleNamespace(get=lambda *a, **k: {}))
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})

import importlib.util
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import types as _types
utils_stub = _types.ModuleType('utils')
utils_stub.FileUtils = types.SimpleNamespace()
saved_utils = sys.modules.get('utils')
sys.modules['utils'] = utils_stub

# load modules dynamically
gi_spec = importlib.util.spec_from_file_location('graph.graph_index', os.path.join(base, 'graph', 'graph_index.py'))
gi_mod = importlib.util.module_from_spec(gi_spec)
gi_spec.loader.exec_module(gi_mod)

retr_spec = importlib.util.spec_from_file_location('graph.enhanced_graph_retriever', os.path.join(base, 'graph', 'enhanced_graph_retriever.py'))
retr_mod = importlib.util.module_from_spec(retr_spec)
retr_spec.loader.exec_module(retr_mod)

if saved_utils is None:
    del sys.modules['utils']
else:
    sys.modules['utils'] = saved_utils

if saved_config is None:
    del sys.modules['config']
else:
    sys.modules['config'] = saved_config

GraphIndex = gi_mod.GraphIndex
EnhancedGraphRetriever = retr_mod.EnhancedGraphRetriever

class DynamicPathScoreTestCase(unittest.TestCase):
    def test_weak_edge_paths_returned(self):
        notes = [
            {"note_id": "n1", "content": "A"},
            {"note_id": "n2", "content": "B"},
        ]
        g = nx.Graph()
        g.add_edge('n1', 'n2', weight=0.05, relation_type='semantic_similarity', reasoning_value=0.05)
        embeddings = np.eye(2)
        index = GraphIndex()
        from unittest.mock import patch
        with patch('networkx.pagerank', return_value={'n1':1.0,'n2':1.0}):
            index.build_index(g, notes, embeddings)

        retriever = EnhancedGraphRetriever(index)
        res = retriever.retrieve_with_reasoning_paths(embeddings[0], top_k=1)
        self.assertTrue(res)
        self.assertTrue(res[0].get('reasoning_paths'))

if __name__ == '__main__':
    unittest.main()
