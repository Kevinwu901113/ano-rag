import unittest
from unittest.mock import patch, MagicMock
import os, sys, types
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules['ollama'] = types.SimpleNamespace(Client=lambda *a, **k: None)

from query.query_processor import QueryProcessor
from config import config

class QueryProcessorTestCase(unittest.TestCase):
    def setUp(self):
        self.notes = [
            {"note_id": "n1", "content": "one"},
            {"note_id": "n2", "content": "two"}
        ]

    @patch('llm.OllamaClient.generate_final_answer', return_value='answer')
    @patch('llm.OllamaClient.evaluate_answer', return_value={'relevance':1})
    @patch('vector_store.embedding_manager.EmbeddingManager._load_local_model')
    @patch('vector_store.embedding_manager.EmbeddingManager.encode_texts', return_value=np.zeros((1,1)))
    @patch('vector_store.VectorRetriever.search')
    @patch('graph.enhanced_graph_retriever.EnhancedGraphRetriever.retrieve_with_reasoning_paths')
    @patch('utils.context_scheduler.MultiHopContextScheduler.schedule_for_multi_hop')
    def test_multi_hop_enabled(self, mock_sched, mock_retrieve, mock_search, m_enc, m_load, m_eval, m_gen):
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

    @patch('llm.OllamaClient.generate_final_answer', return_value='answer')
    @patch('llm.OllamaClient.evaluate_answer', return_value={'relevance':1})
    @patch('vector_store.embedding_manager.EmbeddingManager._load_local_model')
    @patch('vector_store.embedding_manager.EmbeddingManager.encode_texts', return_value=np.zeros((1,1)))
    @patch('vector_store.VectorRetriever.search')
    @patch('graph.graph_retriever.GraphRetriever.retrieve')
    @patch('utils.context_scheduler.ContextScheduler.schedule')
    def test_single_hop(self, mock_sched, mock_retrieve, mock_search, m_enc, m_load, m_eval, m_gen):
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

if __name__ == '__main__':
    unittest.main()
