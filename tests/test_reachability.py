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
def _cfg(key, default=None):
    if key == 'multi_hop':
        return {'path_diversity_threshold': 0}
    return default if default is not None else {}
sys.modules['config'] = types.SimpleNamespace(config=types.SimpleNamespace(get=_cfg))
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})

import importlib.util
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# minimal utils stub to satisfy imports
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

class ReachabilityTestCase(unittest.TestCase):
    def test_answer_reachable_with_short_path(self):
        notes = [
            {"note_id": "q", "content": "query"},
            {"note_id": "m", "content": "middle"},
            {"note_id": "a", "content": "answer"},
        ]
        g = nx.Graph()
        for n in notes:
            g.add_node(n['note_id'], **n)
        g.add_edge('q', 'm', weight=1.0, relation_type='definition', reasoning_value=1.0)
        g.add_edge('m', 'a', weight=1.0, relation_type='definition', reasoning_value=1.0)
        g.add_edge('q', 'a', weight=1.0, relation_type='definition', reasoning_value=1.0)
        embeddings = np.eye(3)
        index = GraphIndex()
        from unittest.mock import patch
        with patch('networkx.pagerank', return_value={'q':1.0,'m':1.0,'a':1.0}):
            index.build_index(g, notes, embeddings)

        retriever = EnhancedGraphRetriever(index)
        results = retriever.retrieve_with_reasoning_paths(embeddings[0], top_k=3)
        answer = next((n for n in results if n.get('note_id') == 'a'), None)
        self.assertIsNotNone(answer)
        paths = answer.get('reasoning_paths', [])
        found = False
        for p in paths:
            if 'q' in p['path'] and 'a' in p['path']:
                if abs(p['path'].index('q') - p['path'].index('a')) <= 2:
                    found = True
                    break
        
        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()
