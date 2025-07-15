import unittest
import os, sys, types, importlib.util
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stub logger
sys.modules['loguru'] = types.SimpleNamespace(
    logger=types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
)

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location('graph.graph_quality', os.path.join(base, 'graph', 'graph_quality.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
compute_metrics = mod.compute_metrics

class GraphQualityMetricsTest(unittest.TestCase):
    def test_compute_metrics(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        g.add_node('c')

        metrics = compute_metrics(g)
        self.assertAlmostEqual(metrics['average_degree'], 2/3)
        self.assertEqual(metrics['num_components'], 2)
        self.assertCountEqual(metrics['component_sizes'], [2, 1])
        self.assertAlmostEqual(metrics['average_shortest_path_length'], 1.0)

if __name__ == '__main__':
    unittest.main()
