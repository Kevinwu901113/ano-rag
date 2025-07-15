import unittest
import os, sys, types
import importlib.util
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub loguru logger
sys.modules['loguru'] = types.SimpleNamespace(
    logger=types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
)

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location('utils.text_utils', os.path.join(base, 'utils', 'text_utils.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TextUtils = mod.TextUtils

class EntityStopwordTestCase(unittest.TestCase):
    def test_stopword_blacklist(self):
        text = "The Apple Company was Created After 1976."
        entities = TextUtils.extract_entities(text)
        self.assertEqual(set(entities), {"Apple Company"})

if __name__ == '__main__':
    unittest.main()
