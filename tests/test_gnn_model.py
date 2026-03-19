"""Tests for ConnectomeGNN."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from conftest import make_mock_graph, make_mock_gnn
from celegans.gnn_model import ConnectomeGNN


class TestGNNModel(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph()
        self.gnn = make_mock_gnn(self.graph)

    def test_forward_output_shape(self):
        out = self.gnn(self.graph.data)
        self.assertEqual(out.shape, (self.graph.data.num_nodes, 1))

    def test_sensory_injection_changes_output(self):
        n_s = len(self.graph.get_sensory_indices())
        if n_s == 0:
            self.skipTest("No sensory neurons in mock graph")
        s_a = np.zeros(n_s, dtype=np.float32)
        s_b = np.ones(n_s, dtype=np.float32)
        out_a = self.gnn(self.graph.data, s_a)
        out_b = self.gnn(self.graph.data, s_b)
        self.assertFalse(np.allclose(out_a, out_b))

    def test_motor_slice(self):
        out = self.gnn(self.graph.data)
        motor_act = self.gnn.get_motor_activations(out)
        n_motor = len(self.graph.get_motor_indices())
        if n_motor == 0:
            self.assertEqual(motor_act.shape, out.shape)
        else:
            self.assertEqual(motor_act.shape[0], n_motor)

    def test_no_nan_in_output(self):
        n_s = max(len(self.graph.get_sensory_indices()), 1)
        sensory = np.random.rand(n_s).astype(np.float32)
        out = self.gnn(self.graph.data, sensory)
        self.assertFalse(np.isnan(out).any(), "NaN in GNN output")
        self.assertFalse(np.isinf(out).any(), "Inf in GNN output")

    def test_no_nan_without_sensory(self):
        out = self.gnn(self.graph.data)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())

    def test_attention_weights_none_for_sage(self):
        _ = self.gnn(self.graph.data)
        self.assertIsNone(self.gnn.get_attention_weights())

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            ConnectomeGNN(input_dim=6, hidden_dim=8, num_layers=0)

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            ConnectomeGNN(input_dim=6, hidden_dim=8, dropout=1.0)

    def test_eval_returns_self(self):
        result = self.gnn.eval()
        self.assertIs(result, self.gnn)


if __name__ == "__main__":
    unittest.main()
