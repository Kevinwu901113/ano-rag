
import unittest
import numpy as np

def calculate_ndcg(retrieved_items, gold_set, k):
    """
    Copy of the function from scripts/evaluate_musique_experiment_3.py
    Note: In MuSiQue, retrieved_items are (title, index) usually, 
    but gold_set is just set of titles.
    """
    relevance = []
    seen_gold = set()
    for i in range(min(k, len(retrieved_items))):
        # retrieved_items are (title, index) usually
        title = retrieved_items[i][0]
        # In MuSiQue, supporting facts are identified by title
        # Only count the first occurrence of a relevant title
        if title in gold_set and title not in seen_gold:
            relevance.append(1)
            seen_gold.add(title)
        else:
            relevance.append(0)
    
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        
    num_gold = len(gold_set)
    ideal_k = min(num_gold, k)
    
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_recall(retrieved_items, gold_set, k):
    """
    Copy of the function from scripts/evaluate_musique_experiment_3.py
    """
    if not gold_set:
        return 0.0
    
    retrieved_k = [x[0] for x in retrieved_items[:k]] # just titles
    # Use set intersection to avoid double counting same document retrieved multiple times
    hits = len(set(retrieved_k) & gold_set)
    return hits / len(gold_set)

class TestMetrics(unittest.TestCase):
    def test_recall_duplicates(self):
        # Gold: DocA, DocB
        gold_set = {"DocA", "DocB"}
        
        # Retrieved: DocA(chunk1), DocA(chunk2), DocC
        retrieved = [("DocA", 1), ("DocA", 2), ("DocC", 3)]
        
        # Recall@3 should be 1/2 = 0.5 (only DocA found, DocB missing)
        # Even though DocA appears twice, it counts as 1 hit.
        r = calculate_recall(retrieved, gold_set, 3)
        self.assertEqual(r, 0.5)

    def test_recall_top_k(self):
        gold_set = {"DocA", "DocB"}
        retrieved = [("DocC", 1), ("DocA", 2), ("DocB", 3)]
        
        # Recall@1: 0 (DocC is not relevant)
        self.assertEqual(calculate_recall(retrieved, gold_set, 1), 0.0)
        # Recall@2: 0.5 (DocA found)
        self.assertEqual(calculate_recall(retrieved, gold_set, 2), 0.5)
        # Recall@3: 1.0 (DocA and DocB found)
        self.assertEqual(calculate_recall(retrieved, gold_set, 3), 1.0)

    def test_ndcg_duplicates(self):
        # Gold: DocA, DocB
        gold_set = {"DocA", "DocB"}
        
        # Retrieved: DocA(chunk1), DocA(chunk2), DocB(chunk1)
        retrieved = [("DocA", 1), ("DocA", 2), ("DocB", 3)]
        
        # Ideal: DocA, DocB (score 1, 1) -> IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # Actual: 
        # Rank 1: DocA -> Rel=1 (New)
        # Rank 2: DocA -> Rel=0 (Duplicate)
        # Rank 3: DocB -> Rel=1 (New)
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # NDCG = 1.5 / 1.6309 = 0.9197
        
        ndcg = calculate_ndcg(retrieved, gold_set, 3)
        
        idcg = 1.0 + 1.0/np.log2(3)
        dcg = 1.0 + 0.0 + 1.0/np.log2(4)
        expected = dcg / idcg
        
        self.assertAlmostEqual(ndcg, expected, places=4)

    def test_ndcg_top_k(self):
        gold_set = {"DocA"}
        retrieved = [("DocB", 1), ("DocA", 2)]
        
        # NDCG@1: Top is DocB (irrelevant). DCG=0. IDCG=1. Result=0.
        self.assertEqual(calculate_ndcg(retrieved, gold_set, 1), 0.0)
        
        # NDCG@2: 
        # Rank 1: DocB -> 0
        # Rank 2: DocA -> 1
        # DCG = 0 + 1/log2(3) = 0.6309
        # IDCG = 1/log2(2) = 1
        # NDCG = 0.6309
        self.assertAlmostEqual(calculate_ndcg(retrieved, gold_set, 2), 1.0/np.log2(3), places=4)

if __name__ == '__main__':
    unittest.main()
