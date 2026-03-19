"""Tests for connectome loading and manipulation."""
import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from conftest import make_mock_graph
from celegans.connectome import build_mock_connectome


class TestConnectome(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph(n_nodes=10, n_edges=20, seed=0)

    def test_load_graph_shape(self):
        self.assertEqual(self.graph.data.num_nodes, 10)
        self.assertEqual(self.graph.data.num_edges, 20)
        self.assertEqual(self.graph.data.x.shape[0], 10)
        self.assertGreater(self.graph.data.x.shape[1], 0)
        self.assertEqual(self.graph.data.edge_index.shape[0], 2)
        self.assertEqual(self.graph.data.edge_index.shape[1], 20)

    def test_neuron_type_assignment(self):
        s = set(self.graph.get_sensory_indices())
        m = set(self.graph.get_motor_indices())
        i = set(self.graph.get_interneuron_indices())
        n = self.graph.data.num_nodes
        self.assertTrue(all(0 <= idx < n for idx in s | m | i))
        self.assertEqual(len(s & m), 0)
        self.assertEqual(len(s & i), 0)
        self.assertEqual(len(m & i), 0)

    def test_ablate_neurons_removes_edges(self):
        neuron = self.graph.node_names[0]
        ablated = self.graph.ablate_neurons([neuron])
        self.assertEqual(ablated.data.num_nodes, self.graph.data.num_nodes - 1)
        self.assertNotIn(neuron, ablated.node_names)
        self.assertLessEqual(ablated.data.num_edges, self.graph.data.num_edges)

    def test_ablate_preserves_original(self):
        orig_nodes = self.graph.data.num_nodes
        orig_edges = self.graph.data.num_edges
        orig_names = list(self.graph.node_names)
        _ = self.graph.ablate_neurons([self.graph.node_names[0]])
        self.assertEqual(self.graph.data.num_nodes, orig_nodes)
        self.assertEqual(self.graph.data.num_edges, orig_edges)
        self.assertEqual(self.graph.node_names, orig_names)

    def test_random_ablation_reproducibility(self):
        g1 = self.graph.ablate_random_synapses(fraction=0.3, seed=99)
        g2 = self.graph.ablate_random_synapses(fraction=0.3, seed=99)
        self.assertEqual(g1.data.num_edges, g2.data.num_edges)
        np.testing.assert_array_equal(g1.data.edge_index, g2.data.edge_index)

    def test_summary_keys(self):
        s = self.graph.summary()
        for key in ("num_nodes", "num_edges", "density", "avg_degree"):
            self.assertIn(key, s)

    def test_ablate_fraction_zero(self):
        g0 = self.graph.ablate_random_synapses(0.0, seed=0)
        self.assertEqual(g0.data.num_edges, self.graph.data.num_edges)

    def test_ablate_fraction_one(self):
        g1 = self.graph.ablate_random_synapses(1.0, seed=0)
        self.assertEqual(g1.data.num_edges, 0)

    def test_different_seeds_produce_valid_graphs(self):
        g1 = build_mock_connectome(n_nodes=10, n_edges=20, seed=1)
        g2 = build_mock_connectome(n_nodes=10, n_edges=20, seed=2)
        self.assertEqual(g1.data.num_edges, 20)
        self.assertEqual(g2.data.num_edges, 20)


if __name__ == "__main__":
    unittest.main()
