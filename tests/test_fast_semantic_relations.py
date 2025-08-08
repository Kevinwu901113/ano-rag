import os
import sys
import importlib.util
import pytest

import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from config import config

utils_stub = types.ModuleType('utils')

class _Dummy:
    pass

class _Batch:
    def __init__(self, *args, **kwargs):
        pass

utils_stub.TextUtils = _Dummy
utils_stub.GPUUtils = _Dummy
utils_stub.BatchProcessor = _Batch
utils_stub.extract_json_from_response = lambda x: x
sys.modules['utils'] = utils_stub

llm_stub = types.ModuleType('llm')
class _LLM:
    def generate(self, prompt):
        return ""
llm_stub.LocalLLM = _LLM
sys.modules['llm'] = llm_stub

MODULE_PATH = os.path.join(ROOT_DIR, 'graph', 'enhanced_relation_extractor.py')
spec = importlib.util.spec_from_file_location('enhanced_relation_extractor', MODULE_PATH)
ere_module = importlib.util.module_from_spec(spec)
sys.modules['enhanced_relation_extractor'] = ere_module
spec.loader.exec_module(ere_module)
EnhancedRelationExtractor = ere_module.EnhancedRelationExtractor

class DummyModel:
    def encode(self, sentences, convert_to_numpy=True):
        # return identical embeddings to yield high similarity
        return [[1.0, 0.0] for _ in sentences]

class DummyLLM:
    def generate(self, prompt):
        return ""


def test_fast_model_produces_relations(monkeypatch):
    cfg = config.load_config()
    cfg['multi_hop']['llm_relation_extraction']['use_fast_model'] = True
    cfg['multi_hop']['llm_relation_extraction']['enabled'] = True

    monkeypatch.setattr(
        'enhanced_relation_extractor.SentenceTransformer',
        lambda *args, **kwargs: DummyModel()
    )
    monkeypatch.setattr(
        'enhanced_relation_extractor.LocalLLM',
        lambda *args, **kwargs: DummyLLM()
    )

    extractor = EnhancedRelationExtractor()
    notes = [
        {
            'note_id': '1',
            'content': 'Cats are small animals.',
            'keywords': ['cats', 'animals'],
            'entities': ['cat'],
            'topic': 'cats'
        },
        {
            'note_id': '2',
            'content': 'Cats are small animals that purr.',
            'keywords': ['cats', 'animals'],
            'entities': ['cat'],
            'topic': 'cats'
        },
    ]

    relations = extractor._extract_semantic_relations_fast(notes)
    assert relations
    r = relations[0]
    assert r['source_id'] == '1'
    assert r['target_id'] == '2'
    assert r['relation_type'] in ('support', 'comparison')


def test_config_toggle_switches_methods(monkeypatch):
    cfg = config.load_config()
    cfg['multi_hop']['llm_relation_extraction']['enabled'] = True

    calls = {'fast': 0, 'llm': 0}

    def fake_fast(self, notes):
        calls['fast'] += 1
        return []

    def fake_llm(self, notes):
        calls['llm'] += 1
        return []

    monkeypatch.setattr(EnhancedRelationExtractor, '_extract_semantic_relations_fast', fake_fast)
    monkeypatch.setattr(EnhancedRelationExtractor, '_extract_semantic_relations_with_llm', fake_llm)
    monkeypatch.setattr(
        'enhanced_relation_extractor.LocalLLM',
        lambda *args, **kwargs: DummyLLM()
    )

    cfg['multi_hop']['llm_relation_extraction']['use_fast_model'] = True
    extractor = EnhancedRelationExtractor()
    extractor.extract_all_relations([
        {'note_id': '1', 'content': 'a', 'keywords': ['a'], 'entities': []},
        {'note_id': '2', 'content': 'b', 'keywords': ['b'], 'entities': []},
    ])
    assert calls['fast'] == 1
    assert calls['llm'] == 0

    cfg['multi_hop']['llm_relation_extraction']['use_fast_model'] = False
    extractor2 = EnhancedRelationExtractor()
    extractor2.extract_all_relations([
        {'note_id': '1', 'content': 'a', 'keywords': ['a'], 'entities': []},
        {'note_id': '2', 'content': 'b', 'keywords': ['b'], 'entities': []},
    ])
    assert calls['fast'] == 1
    assert calls['llm'] == 1
