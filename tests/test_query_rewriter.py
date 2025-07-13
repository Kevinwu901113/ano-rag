import unittest
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub heavy modules
sys.modules['loguru'] = types.SimpleNamespace(logger=types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
))
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules['transformers'] = types.SimpleNamespace(AutoTokenizer=object, AutoModelForCausalLM=object, pipeline=lambda *a, **k: None)
sys.modules['requests'] = types.SimpleNamespace()
sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None)
sys.modules['faiss'] = types.SimpleNamespace()

# provide a dummy LocalLLM before importing QueryRewriter
import types as _types
llm_stub = _types.ModuleType('llm.local_llm')
class DummyLLM:
    def generate(self, *a, **k):
        raise Exception('LLM should not be called')
llm_stub.LocalLLM = DummyLLM
sys.modules['llm.local_llm'] = llm_stub

from config import config
from llm.query_rewriter import QueryRewriter

class QueryRewriterTestCase(unittest.TestCase):
    def setUp(self):
        config.update_config({'query': {'placeholder_split': True}})
        self.rewriter = QueryRewriter()
        self.rewriter._optimize_single_query = lambda q: [q]

    def test_rule_based_split(self):
        self.rewriter._analyze_query = lambda q: {
            'query_type': 'factual',
            'complexity': 'medium',
            'is_multi_question': True,
            'enhancements': []
        }
        self.rewriter._llm_split_multi_queries = lambda q: (_ for _ in ()).throw(AssertionError('LLM called'))
        result = self.rewriter.rewrite_query("What is the capital of France's largest region?")
        self.assertEqual(result['rewritten_queries'][0], "What is France's largest region?")
        self.assertIn('<result_of_q1>', result['rewritten_queries'][1])

    def test_single_hop(self):
        self.rewriter._analyze_query = lambda q: {
            'query_type': 'factual',
            'complexity': 'medium',
            'is_multi_question': False,
            'enhancements': []
        }
        result = self.rewriter.rewrite_query('Who wrote Hamlet?')
        self.assertEqual(result['rewritten_queries'], ['Who wrote Hamlet?'])

if __name__ == '__main__':
    unittest.main()
