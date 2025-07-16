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
        return '{"content": "Recovered text"...'

class TruncatedJsonRecoveryTestCase(unittest.TestCase):
    def test_recover_truncated_json(self):
        gen = AtomicNoteGenerator(llm=DummyLLM())
        sys_prompt = gen._get_atomic_note_system_prompt()
        note = gen._generate_single_atomic_note({'text': 'foo'}, sys_prompt)
        self.assertEqual(note['content'], 'Recovered text')
        self.assertEqual(note['summary'], 'Recovered text')

if __name__ == '__main__':
    unittest.main()
