import unittest
import sys
import os

sys.path.append(os.getcwd())

from utils.output_eval import extract_final_answer, has_final_tag

class TestOutputProtocol(unittest.TestCase):
    def test_basic_final(self):
        text = "Reasoning...\nFINAL: 42"
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "42")

    def test_case_insensitive(self):
        text = "Reasoning...\nFinal: 42"
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "42")
        
        text = "Reasoning...\nfinal: 42"
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "42")

    def test_whitespace_tolerance(self):
        text = "Reasoning...\n   FINAL  :   42   "
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "42")

    def test_missing_final_tag(self):
        text = "Reasoning...\nJust 42"
        self.assertFalse(has_final_tag(text))
        # Fallback to last line behavior per current implementation
        self.assertEqual(extract_final_answer(text), "Just 42")

    def test_multiple_final_tags(self):
        # Should pick the last one
        text = "FINAL: First\nReasoning\nFINAL: Second"
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "Second")

    def test_multiline_answer(self):
        # Current logic only takes the line starting with FINAL:
        text = "FINAL: The answer is\n42"
        self.assertTrue(has_final_tag(text))
        self.assertEqual(extract_final_answer(text), "The answer is")

if __name__ == '__main__':
    unittest.main()
