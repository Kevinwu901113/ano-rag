import unittest
import os, sys, types
import numpy as np
import networkx as nx
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

from query.query_processor import QueryProcessor
from graph.graph_index import GraphIndex
from config import config

TEST_NOTES = [
    {"note_id": "A", "content": "Character B is voiced by John Doe.", "entities": ["B", "John Doe"]},
    {"note_id": "C", "content": "John Doe is married to Jane Smith.", "entities": ["John Doe", "Jane Smith"]},
]

TEST_GRAPH = nx.Graph()
TEST_GRAPH.add_node('A', **TEST_NOTES[0])
TEST_GRAPH.add_node('C', **TEST_NOTES[1])
TEST_GRAPH.add_edge('A', 'C', relation_type='personal_relation', weight=1.0)


def fake_build_index(self, graph, atomic_notes=None, embeddings=None):
    self.graph = graph
    if atomic_notes is not None:
        self.note_id_to_index = {n['note_id']: i for i, n in enumerate(atomic_notes)}
        self.index_to_note_id = {i: n['note_id'] for i, n in enumerate(atomic_notes)}
    if embeddings is not None:
        self.embeddings = embeddings
    self.node_centrality = {n: 1.0 for n in graph.nodes}


class VoiceActorSpouseTestCase(unittest.TestCase):
    @patch('graph.graph_builder.GraphBuilder.build_graph', return_value=TEST_GRAPH)
    @patch('graph.graph_index.GraphIndex.build_index', new=fake_build_index)
    @patch('llm.query_rewriter.QueryRewriter.rewrite_query', return_value={
        'original_query': "Who is the spouse of B's voice actor?",
        'rewritten_queries': ["Who is the spouse of B's voice actor?"],
        'query_type': 'factual',
        'complexity': 'medium',
        'is_multi_question': False,
        'enhancements': []
    })
    @patch('llm.OllamaClient.generate_final_answer', return_value='Jane Smith is the spouse of John Doe.')
    @patch('llm.OllamaClient.evaluate_answer', return_value={'relevance':1})
    @patch('vector_store.embedding_manager.EmbeddingManager._load_local_model')
    @patch('vector_store.embedding_manager.EmbeddingManager.encode_texts', return_value=np.zeros((1,1)))
    @patch('vector_store.VectorRetriever.search')
    @patch('graph.graph_retriever.GraphRetriever.retrieve')
    @patch('utils.context_scheduler.ContextScheduler.schedule')
    def test_voice_actor_spouse(self, mock_sched, mock_graph_ret, mock_search, m_enc, m_load, m_eval, m_gen, m_rewrite, m_build_graph):
        mock_search.return_value = [[TEST_NOTES[0]]]
        mock_graph_ret.return_value = [TEST_NOTES[1]]
        mock_sched.side_effect = lambda notes: notes
        m_load.return_value = None
        config.update_config({'multi_hop':{'enabled':False}})

        processor = QueryProcessor(TEST_NOTES)
        result = processor.process("Who is the spouse of B's voice actor?")
        note_ids = [n['note_id'] for n in result['notes']]
        self.assertIn('A', note_ids)
        self.assertIn('C', note_ids)
        self.assertIn('Jane Smith', result['answer'])


if __name__ == '__main__':
    unittest.main()
