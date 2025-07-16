import unittest
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub heavy modules
sys.modules['ollama'] = types.SimpleNamespace(Client=lambda *a, **k: None)
sys.modules['loguru'] = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None,
                                                                            debug=lambda *a, **k: None,
                                                                            warning=lambda *a, **k: None,
                                                                            error=lambda *a, **k: None))
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **k: x)
sys.modules['jsonlines'] = types.SimpleNamespace()
sys.modules['docx'] = types.SimpleNamespace(Document=lambda *a, **k: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})
sys.modules['transformers'] = types.SimpleNamespace(AutoTokenizer=object, AutoModelForCausalLM=object, pipeline=lambda *a, **k: None)
sys.modules['requests'] = types.SimpleNamespace()
sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None)
sys.modules['faiss'] = types.SimpleNamespace()

import importlib.util
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
chunk_spec = importlib.util.spec_from_file_location('doc.chunker', os.path.join(base, 'doc', 'chunker.py'))
chunk_mod = importlib.util.module_from_spec(chunk_spec)
chunk_spec.loader.exec_module(chunk_mod)
DocumentChunker = chunk_mod.DocumentChunker

from unittest.mock import patch

class MusiqueChunkTestCase(unittest.TestCase):
    def test_question_excluded_by_default(self):
        data = {
            "paragraphs": [
                {"paragraph_text": "para one", "idx": 0},
                {"paragraph_text": "para two", "idx": 1}
            ],
            "question": "Who did it?"
        }
        chunker = DocumentChunker()
        with patch.object(DocumentChunker, '_read_document_content', return_value=data):
            chunks = chunker.chunk_document('dummy.json', {
                'file_path': 'dummy.json',
                'file_name': 'dummy.json',
                'file_hash': '0'
            })
        self.assertTrue(chunks)
        for ch in chunks:
            self.assertNotIn('Who did it?', ch['text'])

if __name__ == '__main__':
    unittest.main()
