import unittest
import os, sys, types
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub heavy modules
sys.modules['ollama'] = types.SimpleNamespace(Client=lambda *a, **k: None)
sys.modules['loguru'] = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **k: x)
sys.modules['jsonlines'] = types.SimpleNamespace()
sys.modules['docx'] = types.SimpleNamespace(Document=lambda *a, **k: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})
sys.modules['transformers'] = types.SimpleNamespace(AutoTokenizer=object, AutoModelForCausalLM=object, pipeline=lambda *a, **k: None)
sys.modules['requests'] = types.SimpleNamespace()
sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None)
sys.modules['faiss'] = types.SimpleNamespace()

from config import config
from graph.relation_extractor import RelationExtractor
from graph.enhanced_relation_extractor import EnhancedRelationExtractor

class PersonalRelationTestCase(unittest.TestCase):
    def setUp(self):
        self.notes = [
            {"note_id": "n1", "content": "Alice is married to Bob.", "entities": ["Alice", "Bob"]},
            {"note_id": "n2", "content": "Bob likes cooking.", "entities": ["Bob"]},
        ]
        config.update_config({'multi_hop': {'llm_relation_extraction': {'enabled': False}, 'topic_group_llm': {'enabled': False}}})

    def test_base_extractor_spouse(self):
        extractor = RelationExtractor()
        rels = extractor.extract_all_relations(self.notes)
        found = any(r['relation_type'] == 'personal_relation' and {'n1','n2'} == {r['source_id'], r['target_id']} for r in rels)
        self.assertTrue(found)

    def test_enhanced_extractor_spouse(self):
        extractor = EnhancedRelationExtractor()
        rels = extractor.extract_all_relations(self.notes)
        found = any(r['relation_type'] == 'personal_relation' and {'n1','n2'} == {r['source_id'], r['target_id']} for r in rels)
        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()
