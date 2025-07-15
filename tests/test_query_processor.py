import unittest
from unittest.mock import patch, MagicMock
import os, sys, types
import numpy as np
import networkx as nx
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

from query.query_processor import QueryProcessor
from graph.graph_index import GraphIndex
from graph.multi_hop_query_processor import MultiHopQueryProcessor
from graph.graph_retriever import GraphRetriever
from config import config

class QueryProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.notes = [
            {"note_id": "n1", "content": "one"},
            {"note_id": "n2", "content": "two"}
        ]

    @patch('graph.graph_index.GraphIndex.build_index')
    @patch('llm.OllamaClient.generate_final_answer', return_value='answer')
    @patch('llm.OllamaClient.evaluate_answer', return_value={'relevance':1})
    @patch('vector_store.embedding_manager.EmbeddingManager._load_local_model')
    @patch('vector_store.embedding_manager.EmbeddingManager.encode_texts', return_value=np.zeros((1,1)))
    @patch('vector_store.VectorRetriever.search')
    @patch('graph.enhanced_graph_retriever.EnhancedGraphRetriever.retrieve_with_reasoning_paths')
    @patch('utils.context_scheduler.MultiHopContextScheduler.schedule_for_multi_hop')
    def test_multi_hop_enabled(self, mock_sched, mock_retrieve, mock_search, m_enc, m_load, m_eval, m_gen, mock_index):
        mock_search.return_value = [[{"note_id":"n1","content":"one","retrieval_info":{"similarity":1}}]]
        mock_retrieve.return_value = [{"note_id":"n2","content":"two","reasoning_paths":[{"path":["n1","n2"],"path_score":0.9}]}]
        mock_sched.side_effect = lambda notes, paths: notes
        m_load.return_value = None
        config.update_config({'multi_hop':{'enabled':True}})

        processor = QueryProcessor(self.notes)
        res = processor.process('test')
        self.assertTrue(processor.multi_hop_enabled)
        self.assertIn('reasoning', res)
        self.assertTrue(res['reasoning'])
        self.assertIn('reasoning_paths', res['notes'][1])

    @patch('graph.graph_index.GraphIndex.build_index')
    @patch('llm.OllamaClient.generate_final_answer', return_value='answer')
    @patch('llm.OllamaClient.evaluate_answer', return_value={'relevance':1})
    @patch('vector_store.embedding_manager.EmbeddingManager._load_local_model')
    @patch('vector_store.embedding_manager.EmbeddingManager.encode_texts', return_value=np.zeros((1,1)))
    @patch('vector_store.VectorRetriever.search')
    @patch('graph.graph_retriever.GraphRetriever.retrieve')
    @patch('utils.context_scheduler.ContextScheduler.schedule')
    def test_single_hop(self, mock_sched, mock_retrieve, mock_search, m_enc, m_load, m_eval, m_gen, mock_index):
        mock_search.return_value = [[{"note_id":"n1","content":"one","retrieval_info":{"similarity":1}}]]
        mock_retrieve.return_value = [{"note_id":"n2","content":"two"}]
        mock_sched.side_effect = lambda notes: notes
        m_load.return_value = None
        config.update_config({'multi_hop':{'enabled':False}})

        processor = QueryProcessor(self.notes)
        res = processor.process('test')
        self.assertFalse(processor.multi_hop_enabled)
        self.assertIsNone(res['reasoning'])
        self.assertNotIn('reasoning_paths', res['notes'][0])

    @patch('graph.multi_hop_query_processor.GraphBuilder')
    @patch('graph.enhanced_graph_retriever.logger')
    def test_multi_hop_initial_candidates(self, mock_logger, mock_builder):
        g = nx.Graph()
        g.add_edge('n1', 'n2', weight=1.0)
        embeddings = np.eye(2)
        index = GraphIndex()
        from unittest.mock import patch
        with patch('networkx.pagerank', return_value={'n1':1.0,'n2':1.0}):
            index.build_index(g, self.notes, embeddings)
        mock_builder.return_value = MagicMock()
        proc = MultiHopQueryProcessor(self.notes, embeddings, graph_index=index)
        mock_logger.warning = MagicMock()
        res = proc.retrieve(embeddings[0])
        self.assertTrue(res['notes'])
        mock_logger.warning.assert_not_called()

    def test_importance_score_affects_retrieval(self):
        notes = [
            {"note_id": "n1", "content": "seed", "importance_score": 1.0},
            {"note_id": "n2", "content": "two", "importance_score": 1.0},
            {"note_id": "n3", "content": "three", "importance_score": 0.1},
        ]
        g = nx.Graph()
        g.add_node('n1', **notes[0])
        g.add_node('n2', **notes[1])
        g.add_node('n3', **notes[2])
        g.add_edge('n1', 'n2', weight=1.0)
        g.add_edge('n1', 'n3', weight=1.0)
        embeddings = np.eye(3)
        index = GraphIndex()
        with patch('networkx.pagerank', return_value={'n1': 1.0, 'n2': 0.5, 'n3': 0.8}):
            index.build_index(g, notes, embeddings)
        retriever = GraphRetriever(index, k_hop=1)
        res = retriever.retrieve(['n1'])
        ordered = [n['note_id'] for n in res]
        self.assertEqual(ordered[0], 'n2')

if __name__ == '__main__':
    unittest.main()
