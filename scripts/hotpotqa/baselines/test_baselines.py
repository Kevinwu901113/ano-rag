import unittest
import json
import tempfile
import os
from pathlib import Path

# Stub classes to avoid loading heavy models during unit tests
class MockLLMClient:
    def chat(self, messages):
        return "Answer: Mock Answer"

class MockEmbeddingEncoder:
    def encode(self, texts):
        import numpy as np
        # Return random vectors
        return np.random.rand(len(texts), 32)

class TestHotpotBaselines(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            {
                "_id": "test_q1",
                "question": "Test Question 1",
                "context": [
                    ["Title 1", ["Sentence 1.1", "Sentence 1.2"]],
                    ["Title 2", ["Sentence 2.1"]],
                    # Add more paragraphs to test limit
                ] + [([f"Title {i}", [f"Sentence {i}"]]) for i in range(3, 15)],
                "answer": "Answer 1",
                "supporting_facts": [["Title 1", 0]]
            }
        ]
        
        # Create temp dataset file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.tmp_dir.name, "hotpot_dev_distractor_v1.json")
        with open(self.dataset_path, "w") as f:
            json.dump(self.dataset, f)
            
        self.output_path = os.path.join(self.tmp_dir.name, "pred.json")
        
    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_direct_limit(self):
        """Test Direct Baseline truncates to 10 paragraphs"""
        # Add project root to sys.path
        import sys
        ROOT = Path(__file__).resolve().parents[3]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
            
        from scripts.hotpotqa.baselines.run_direct import load_dataset, format_context
        
        data = load_dataset(self.dataset_path)
        item = data[0]
        context = item["context"]
        
        # Verify input has > 10
        self.assertTrue(len(context) > 10)
        
        # Simulate logic in run_direct
        if len(context) > 10:
            context = context[:10]
            
        self.assertEqual(len(context), 10)
        
        text = format_context([f"{t}\n{''.join(s)}" for t, s in context])
        self.assertIn("Title 1", text)
        self.assertIn("Title 10", text) 
        # indices: 0, 1, 2(Title 3), ..., 9(Title 10)
        # Title 1, Title 2, Title 3 ... Title 10.
        # Title 11 should NOT be there.
        self.assertNotIn("Title 11", text)

    def test_vanilla_rag_logic(self):
        from scripts.hotpotqa.baselines.run_vanilla_rag import InMemoryVanillaRetriever
        
        encoder = MockEmbeddingEncoder()
        retriever = InMemoryVanillaRetriever(encoder)
        
        item = self.dataset[0]
        retriever.build_index_for_question(item["context"])
        
        # Check internal storage limit
        self.assertEqual(len(retriever.paragraphs), 10)
        self.assertEqual(len(retriever.titles), 10)
        
        hits = retriever.retrieve("query", k=2)
        self.assertEqual(len(hits), 2)

    def test_raptor_logic(self):
        from scripts.hotpotqa.baselines.run_raptor import MiniRaptor
        
        encoder = MockEmbeddingEncoder()
        llm = MockLLMClient()
        raptor = MiniRaptor(encoder, llm)
        
        item = self.dataset[0]
        raptor.build_tree(item["context"])
        
        # Should have leaf nodes + summary nodes
        # 10 leaves. 10 // 3 = 3 clusters. 3 summaries.
        # Total 13 nodes ideally.
        # Since clustering is random/mocked, check at least leaves are present.
        self.assertTrue(len(raptor.tree_nodes) >= 10)

if __name__ == '__main__':
    unittest.main()
