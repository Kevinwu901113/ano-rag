import unittest
from unittest.mock import patch
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.modules['ollama'] = types.SimpleNamespace(Client=lambda *a, **k: None)
sys.modules['loguru'] = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **k: x)
sys.modules['jsonlines'] = types.SimpleNamespace()
sys.modules['docx'] = types.SimpleNamespace(Document=lambda *a, **k: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})
sys.modules['transformers'] = types.SimpleNamespace(
    AutoTokenizer=object,
    AutoModelForCausalLM=object,
    pipeline=lambda *a, **k: None
)
sys.modules['requests'] = types.SimpleNamespace()
sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None)
sys.modules['faiss'] = types.SimpleNamespace()

from graph.enhanced_relation_extractor import EnhancedRelationExtractor
from config import config

class TopicGroupRelationsTestCase(unittest.TestCase):
    def setUp(self):
        self.notes = [
            {"note_id": "n1", "content": "A1", "cluster_id": 1, "topic": "t"},
            {"note_id": "n2", "content": "A2", "cluster_id": 1, "topic": "t"},
            {"note_id": "n3", "content": "A3", "cluster_id": 1, "topic": "t"},
        ]

    @patch('llm.LocalLLM.generate', return_value='[{"source_index":0,"target_index":1,"relation_type":"causal","strength":0.8,"confidence":0.9}]')
    def test_group_relations_generated(self, mock_gen):
        config.update_config({'multi_hop': {
            'topic_group_llm': {'enabled': True, 'min_group_size': 3, 'max_notes': 5},
            'llm_relation_extraction': {'enabled': False}
        }})

        extractor = EnhancedRelationExtractor()
        rels = extractor.extract_all_relations(self.notes)
        found = any(r['relation_type'] == 'causal' and r['source_id'] == 'n1' and r['target_id'] == 'n2' for r in rels)
        self.assertTrue(found)
        mock_gen.assert_called()

if __name__ == '__main__':
    unittest.main()
