"""Tests for LIF neuron dynamics."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from conftest import make_mock_graph, make_mock_lif
from celegans.lif_neurons import LIFLayer, LIFNeuronBank


class TestLIFLayer(unittest.TestCase):

    def test_spike_on_threshold(self):
        layer = LIFLayer(1, tau_mem=20.0, dt=0.1, threshold=1.0, reset_potential=0.0)
        spk, mem = layer.step(np.array([2.0]))
        self.assertEqual(spk[0], 1.0, "Should have fired above threshold")

    def test_no_spike_below_threshold(self):
        layer = LIFLayer(1, tau_mem=20.0, dt=0.1, threshold=1.0, reset_potential=0.0)
        spk, mem = layer.step(np.array([0.1]))
        self.assertEqual(spk[0], 0.0)

    def test_reset_after_spike(self):
        layer = LIFLayer(1, tau_mem=20.0, dt=0.1, threshold=1.0, reset_potential=0.0)
        spk, mem = layer.step(np.array([5.0]))
        self.assertEqual(spk[0], 1.0)
        self.assertLess(mem[0], 4.0, "Membrane should have reset after spike")

    def test_state_reset(self):
        graph = make_mock_graph()
        bank = make_mock_lif(graph)
        act = np.random.rand(bank.n_total) * 2
        bank.step(act)
        self.assertEqual(len(bank._spike_history), 1)
        bank.reset_state()
        self.assertEqual(len(bank._spike_history), 0)
        self.assertAlmostEqual(bank.sensory_lif.mem.max(), 0.0, places=5)

    def test_spike_history_shape(self):
        graph = make_mock_graph()
        bank = make_mock_lif(graph)
        T, n = 5, bank.n_total
        for _ in range(T):
            bank.step(np.random.rand(n))
        hist = bank.get_spike_history()
        self.assertEqual(hist.shape, (T, n))

    def test_spike_history_binary(self):
        graph = make_mock_graph()
        bank = make_mock_lif(graph)
        n = bank.n_total
        for _ in range(3):
            bank.step(np.random.rand(n) * 3)
        hist = bank.get_spike_history()
        unique = np.unique(hist)
        for v in unique:
            self.assertIn(v, (0.0, 1.0))

    def test_recent_spike_rates_shape(self):
        graph = make_mock_graph()
        bank = make_mock_lif(graph)
        n = bank.n_total
        for _ in range(15):
            bank.step(np.random.rand(n))
        rates = bank.get_recent_spike_rates(window=10)
        self.assertEqual(rates.shape, (n,))
        self.assertTrue((rates >= 0).all() and (rates <= 1).all())

    def test_no_nan_output(self):
        graph = make_mock_graph()
        bank = make_mock_lif(graph)
        n = bank.n_total
        for _ in range(5):
            spk, mem = bank.step(np.random.rand(n, 1) * 2)
            self.assertFalse(np.isnan(spk).any())
            self.assertFalse(np.isnan(mem).any())


if __name__ == "__main__":
    unittest.main()
