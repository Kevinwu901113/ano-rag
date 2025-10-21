import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

mirage_module = types.ModuleType("MIRAGE")
mirage_utils_module = types.ModuleType("MIRAGE.utils")
mirage_utils_module.load_json = lambda path: []
mirage_utils_module.convert_doc_pool = lambda doc_pool: doc_pool
mirage_module.utils = mirage_utils_module
sys.modules.setdefault("MIRAGE", mirage_module)
sys.modules.setdefault("MIRAGE.utils", mirage_utils_module)

from main_mirage import MirageConfig, MirageRunner


class DummyEmbeddingManager:
    def encode_texts(self, texts: List[str]):
        embeddings = []
        for text in texts:
            text_lower = (text or '').lower()
            embeddings.append([
                text_lower.count('alpha'),
                text_lower.count('beta')
            ])
        return np.array(embeddings, dtype=float)


class DummyRetriever:
    def __init__(self):
        self.embedding_manager = DummyEmbeddingManager()


def _create_runner(tmp_path) -> MirageRunner:
    config = MirageConfig(
        run_id='test',
        mode='mixed',
        topk=3,
        new_run=True,
        debug=True,
        dataset_path='',
        doc_pool_path='',
        oracle_path='',
        result_dir=str(tmp_path),
        retriever_type='dense',
        embed_model='',
        rebuild_index=False,
        model_name='model',
        temperature=0.0,
        max_tokens=0,
        seed=None,
        note_engines=[],
        enable_notes=False,
        enable_graph=False,
        max_workers_query=1,
        max_workers_note=1,
        start_time=0.0,
    )
    runner = MirageRunner(config)
    runner.retriever = DummyRetriever()
    return runner


def test_remap_falls_back_to_offsets_on_hash_mismatch(tmp_path):
    runner = _create_runner(tmp_path)
    doc_text = 'Alpha segment content'
    runner.doc_pool = [
        {
            'doc_name': 'DocA',
            'doc_chunk': doc_text,
            'offsets': [0, len(doc_text)],
        }
    ]
    runner._build_doc_pool_indexes()

    retrieval_results = [
        {
            'note_id': 'DocA#0',
            'chunk_id': 'DocA#0',
            'doc_name': 'DocA',
            'doc_hash': 'not-real-hash',
            'offsets': [0, len(doc_text)],
            'content': doc_text,
        }
    ]

    contexts = runner.remap_notes_to_chunks(retrieval_results)

    assert contexts == [{'title': 'DocA', 'text': doc_text}]


def test_remap_uses_offsets_when_available(tmp_path):
    runner = _create_runner(tmp_path)
    doc_text = 'Beta knowledge base'
    runner.doc_pool = [
        {
            'doc_name': 'DocB',
            'doc_chunk': doc_text,
            'offsets': [5, 5 + len(doc_text)],
        }
    ]
    runner._build_doc_pool_indexes()

    retrieval_results = [
        {
            'note_id': 'DocB#0',
            'doc_name': 'DocB',
            'offsets': [5, 5 + len(doc_text)],
            'content': doc_text,
        }
    ]

    contexts = runner.remap_notes_to_chunks(retrieval_results)

    assert contexts == [{'title': 'DocB', 'text': doc_text}]


def test_remap_embedding_fallback(tmp_path):
    runner = _create_runner(tmp_path)
    alpha_text = 'Alpha source content'
    beta_text = 'Beta reference text'
    runner.doc_pool = [
        {
            'doc_name': 'AlphaDoc',
            'doc_chunk': alpha_text,
            'offsets': [0, len(alpha_text)],
        },
        {
            'doc_name': 'BetaDoc',
            'doc_chunk': beta_text,
            'offsets': [0, len(beta_text)],
        },
    ]
    runner._build_doc_pool_indexes()

    retrieval_results = [
        {
            'note_id': 'unknown',
            'content': 'alpha evidence snippet',
        }
    ]

    contexts = runner.remap_notes_to_chunks(retrieval_results)

    assert contexts[0]['title'] == 'AlphaDoc'
    assert contexts[0]['text'] == alpha_text
