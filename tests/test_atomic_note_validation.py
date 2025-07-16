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

gen_spec = importlib.util.spec_from_file_location('llm.atomic_note_generator', os.path.join(base, 'llm', 'atomic_note_generator.py'))
gen_mod = importlib.util.module_from_spec(gen_spec)
gen_spec.loader.exec_module(gen_mod)
AtomicNoteGenerator = gen_mod.AtomicNoteGenerator

class DummyLLM:
    def generate(self, *a, **k):
        return ''

class AtomicNoteValidationTestCase(unittest.TestCase):
    def test_similarity_threshold(self):
        gen = AtomicNoteGenerator(llm=DummyLLM())
        notes = [
            {
                'note_id': '1',
                'content': 'This is a test note',
                'original_text': 'This is a test note in document',
                'importance_score': 0.5
            },
            {
                'note_id': '2',
                'content': 'Short text',
                'original_text': 'Completely different words nowhere same',
                'importance_score': 0.5
            }
        ]
        valid = gen.validate_atomic_notes(notes)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]['note_id'], '1')

if __name__ == '__main__':
    unittest.main()
