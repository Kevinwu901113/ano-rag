import unittest
import os, sys, types
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

import importlib.util
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

chunk_spec = importlib.util.spec_from_file_location('doc.chunker', os.path.join(base, 'doc', 'chunker.py'))
chunk_mod = importlib.util.module_from_spec(chunk_spec)
chunk_spec.loader.exec_module(chunk_mod)
DocumentChunker = chunk_mod.DocumentChunker

gen_spec = importlib.util.spec_from_file_location('llm.atomic_note_generator', os.path.join(base, 'llm', 'atomic_note_generator.py'))
gen_mod = importlib.util.module_from_spec(gen_spec)
gen_spec.loader.exec_module(gen_mod)
AtomicNoteGenerator = gen_mod.AtomicNoteGenerator

class DummyLLM:
    def generate(self, *a, **k):
        return '{"content":"","summary":"","keywords":[],"entities":[],"concepts":[],"importance_score":0.5,"note_type":"fact"}'

class PronounEntityTestCase(unittest.TestCase):
    def test_pronoun_note_keeps_entity(self):
        text = "Alice went to the market. She bought apples."
        chunker = DocumentChunker()
        chunker.chunk_size = 30
        chunks = chunker._chunk_text_content(text, 'dummy.json', {'file_path':'dummy.json', 'file_name':'dummy.json', 'file_hash':'0'})
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[1]['primary_entity'], 'Alice')
        self.assertTrue(chunks[1]['text'].startswith('Alice'))
        gen = AtomicNoteGenerator(llm=DummyLLM())
        # patch tqdm to provide a dummy context manager
        class _Dummy:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
            def update(self, *a, **k):
                pass

        from unittest.mock import patch
        with patch('utils.batch_processor.tqdm', lambda *a, **k: _Dummy()):
            notes = gen.generate_atomic_notes(chunks)
        self.assertIn('Alice', notes[1]['entities'])

if __name__ == '__main__':
    unittest.main()
