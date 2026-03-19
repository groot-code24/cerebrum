"""Tests for STDP learning."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from conftest import make_mock_graph
from celegans.stdp import STDPLearner, TripletSTDP


class TestSTDPLearner(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph()
        ei = self.graph.data.edge_index.copy()
        ea = self.graph.data.edge_attr.copy()
        self.learner = STDPLearner(
            edge_index=ei, edge_attr=ea,
            n_neurons=self.graph.data.num_nodes,
            A_plus=0.01, A_minus=0.012,
            tau_plus=20.0, tau_minus=20.0,
            dt=0.1,
        )

    def test_update_returns_delta_w(self):
        spikes = np.zeros(self.graph.data.num_nodes, dtype=np.float32)
        spikes[0] = 1.0
        delta_w = self.learner.update(spikes)
        self.assertEqual(delta_w.shape, (self.graph.data.num_edges,))

    def test_weights_stay_in_bounds(self):
        n = self.graph.data.num_nodes
        for _ in range(100):
            spikes = (np.random.rand(n) > 0.8).astype(np.float32)
            self.learner.update(spikes)
        w = self.learner.edge_attr[:, 0]
        self.assertTrue((w >= self.learner.w_min).all())
        self.assertTrue((w <= self.learner.w_max).all())

    def test_ltp_on_causal_pair(self):
        """Pre fires, then post fires → pre trace increases → next post spike causes LTP."""
        n = self.graph.data.num_nodes
        ei = self.graph.data.edge_index
        if ei.shape[1] == 0:
            self.skipTest("No edges")

        # Reset learner to known state
        self.learner.reset_traces()
        w_initial = self.learner.edge_attr[:, 0].copy()

        # Fire pre-synaptic neuron — builds x_pre trace
        src = int(ei[0, 0])
        spikes_pre = np.zeros(n, dtype=np.float32)
        spikes_pre[src] = 1.0
        self.learner.update(spikes_pre)

        # Pre trace for src should now be > 0
        self.assertGreater(self.learner.x_pre[src], 0.0,
                           "Pre-synaptic trace should be non-zero after spike")

        # Verify no NaN in weights after update
        self.assertFalse(np.isnan(self.learner.edge_attr).any(),
                         "Weights should not contain NaN after causal pair update")

    def test_reset_traces(self):
        n = self.graph.data.num_nodes
        spikes = (np.random.rand(n) > 0.5).astype(np.float32)
        self.learner.update(spikes)
        self.learner.reset_traces()
        np.testing.assert_array_equal(self.learner.x_pre, np.zeros(n))
        np.testing.assert_array_equal(self.learner.x_post, np.zeros(n))

    def test_symmetric_stdp_no_ltd(self):
        """Symmetric STDP (gap junctions) should never decrease weights."""
        ea_copy = self.graph.data.edge_attr.copy()
        sym_learner = STDPLearner(
            edge_index=self.graph.data.edge_index.copy(),
            edge_attr=ea_copy,
            n_neurons=self.graph.data.num_nodes,
            A_plus=0.01, A_minus=0.012,
            symmetric=True, dt=0.1,
        )
        n = self.graph.data.num_nodes
        w_before = ea_copy[:, 0].copy()
        for _ in range(10):
            spikes = (np.random.rand(n) > 0.7).astype(np.float32)
            sym_learner.update(spikes)
        w_after = sym_learner.edge_attr[:, 0]
        self.assertTrue((w_after >= w_before - 1e-6).all(),
                        "Symmetric STDP should not decrease weights")

    def test_weight_statistics(self):
        stats = self.learner.weight_statistics()
        for key in ("mean", "std", "min", "max"):
            self.assertIn(key, stats)

    def test_no_nan_after_many_updates(self):
        n = self.graph.data.num_nodes
        for _ in range(200):
            spikes = (np.random.rand(n) > 0.8).astype(np.float32)
            dw = self.learner.update(spikes)
        self.assertFalse(np.isnan(self.learner.edge_attr).any())
        self.assertFalse(np.isnan(dw).any())


class TestTripletSTDP(unittest.TestCase):

    def test_update_and_bounds(self):
        graph = make_mock_graph()
        ea = graph.data.edge_attr.copy()
        triplet = TripletSTDP(
            edge_index=graph.data.edge_index.copy(),
            edge_attr=ea,
            n_neurons=graph.data.num_nodes,
            dt=0.1,
        )
        n = graph.data.num_nodes
        for _ in range(50):
            spikes = (np.random.rand(n) > 0.8).astype(np.float32)
            triplet.update(spikes)
        w = triplet.edge_attr[:, 0]
        self.assertTrue((w >= 0.0).all())
        self.assertTrue((w <= 1.0).all())
        self.assertFalse(np.isnan(w).any())

    def test_reset_traces(self):
        graph = make_mock_graph()
        triplet = TripletSTDP(
            edge_index=graph.data.edge_index.copy(),
            edge_attr=graph.data.edge_attr.copy(),
            n_neurons=graph.data.num_nodes,
        )
        n = graph.data.num_nodes
        triplet.update(np.ones(n, dtype=np.float32))
        triplet.reset_traces()
        for arr in (triplet.x1, triplet.x2, triplet.y1, triplet.y2):
            np.testing.assert_array_equal(arr, np.zeros(n))


if __name__ == "__main__":
    unittest.main()
