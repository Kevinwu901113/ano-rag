import unittest
import sys, types, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules['jsonlines'] = types.SimpleNamespace()
sys.modules['docx'] = types.SimpleNamespace(Document=lambda *a, **k: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})
sys.modules['loguru'] = types.SimpleNamespace(
    logger=types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
)
sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules['cudf'] = types.SimpleNamespace()
sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **k: x)
from utils.json_utils import clean_control_characters, extract_json_from_response

class JsonUtilsTestCase(unittest.TestCase):
    def test_clean_control_characters(self):
        text = "a\x00b\x01c\n"
        self.assertEqual(clean_control_characters(text), "abc\n")

    def test_extract_from_codeblock(self):
        resp = "```json\n{\"a\":1}\n```"
        self.assertEqual(extract_json_from_response(resp), '{"a":1}')

    def test_extract_from_surrounding_text(self):
        resp = "prefix {\"b\":2} suffix"
        self.assertEqual(extract_json_from_response(resp), '{"b":2}')

    def test_extract_invalid(self):
        self.assertEqual(extract_json_from_response("nothing here"), "")

if __name__ == '__main__':
    unittest.main()
