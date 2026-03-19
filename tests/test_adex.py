"""Tests for AdEx neuron dynamics."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from conftest import make_mock_graph
from celegans.adex_neurons import AdExLayer, AdExNeuronBank, ADEX_PARAMS


class TestAdExLayer(unittest.TestCase):

    def test_spike_on_large_input(self):
        """Large sustained input should cause a spike within 200 steps.

        With 500 pA input the sensory AdEx neuron takes ~145 Euler steps
        (dt=0.1ms) to charge from E_L=-70mV to V_peak=+20mV.
        """
        layer = AdExLayer(1, neuron_type="sensory", dt=0.1)
        spiked = False
        for _ in range(200):
            spk, _ = layer.step(np.array([500.0]))
            if spk[0] == 1.0:
                spiked = True
                break
        self.assertTrue(spiked, "Should spike within 200 steps of 500 pA sustained input")

    def test_no_spike_on_zero_input(self):
        layer = AdExLayer(1, neuron_type="sensory", dt=0.1)
        spk, mem = layer.step(np.array([0.0]))
        self.assertEqual(spk[0], 0.0, "Should not spike on zero input")

    def test_graded_interneuron_no_spikes(self):
        """Interneurons default to graded (non-spiking) mode."""
        layer = AdExLayer(5, neuron_type="interneuron", dt=0.1)
        for _ in range(20):
            spk, mem = layer.step(np.random.rand(5) * 100)
        self.assertEqual(spk.sum(), 0.0, "Interneurons should not spike")

    def test_membrane_output_in_millivolts(self):
        layer = AdExLayer(1, neuron_type="sensory", dt=0.1)
        _, mem = layer.step(np.array([0.0]))
        # At rest, V ≈ E_L = -70 mV → output in mV range
        self.assertLess(mem[0], 10.0, "Membrane output should be in mV range")
        self.assertGreater(mem[0], -200.0)

    def test_reset_after_spike(self):
        layer = AdExLayer(1, neuron_type="motor", dt=0.1)
        spk, mem = layer.step(np.array([100.0]))  # force spike
        if spk[0] == 1.0:
            # V should reset to near V_r
            V_r_mV = ADEX_PARAMS["motor"]["V_r"] * 1e3
            self.assertLess(mem[0], 0.0, "Membrane should reset below 0 after spike")

    def test_reset_state_clears_adaptation(self):
        layer = AdExLayer(3, neuron_type="motor", dt=0.1)
        for _ in range(10):
            layer.step(np.random.rand(3) * 20)
        layer.reset_state()
        # w should be zero after reset
        np.testing.assert_array_equal(layer.w, np.zeros(3))

    def test_output_shapes(self):
        n = 10
        layer = AdExLayer(n, neuron_type="sensory", dt=0.1)
        spk, mem = layer.step(np.random.rand(n))
        self.assertEqual(spk.shape, (n,))
        self.assertEqual(mem.shape, (n,))
        self.assertEqual(spk.dtype, np.float32)
        self.assertEqual(mem.dtype, np.float32)

    def test_no_nan(self):
        layer = AdExLayer(10, neuron_type="sensory", dt=0.1)
        for _ in range(20):
            spk, mem = layer.step(np.random.rand(10) * 50)
        self.assertFalse(np.isnan(spk).any())
        self.assertFalse(np.isnan(mem).any())

    def test_graded_output_range(self):
        layer = AdExLayer(5, neuron_type="interneuron", dt=0.1)
        layer.step(np.random.rand(5) * 10)
        graded = layer.get_graded_output()
        self.assertTrue((graded >= 0.0).all())
        self.assertTrue((graded <= 1.0).all())

    def test_invalid_neuron_type(self):
        with self.assertRaises(ValueError):
            AdExLayer(1, neuron_type="alien")


class TestAdExNeuronBank(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph()
        n = self.graph.data.num_nodes
        self.bank = AdExNeuronBank(
            n_total=n,
            sensory_indices=self.graph.get_sensory_indices(),
            interneuron_indices=self.graph.get_interneuron_indices(),
            motor_indices=self.graph.get_motor_indices(),
            dt=0.1,
        )

    def test_step_output_shape(self):
        n = self.bank.n_total
        spk, mem = self.bank.step(np.random.rand(n))
        self.assertEqual(spk.shape, (n,))
        self.assertEqual(mem.shape, (n,))

    def test_spike_history_accumulates(self):
        n = self.bank.n_total
        T = 5
        for _ in range(T):
            self.bank.step(np.random.rand(n))
        hist = self.bank.get_spike_history()
        self.assertEqual(hist.shape[0], T)
        self.assertEqual(hist.shape[1], n)

    def test_voltage_history(self):
        n = self.bank.n_total
        for _ in range(3):
            self.bank.step(np.random.rand(n) * 5)
        vh = self.bank.get_voltage_history()
        self.assertEqual(vh.shape, (3, n))

    def test_reset_clears_history(self):
        n = self.bank.n_total
        self.bank.step(np.random.rand(n))
        self.bank.reset_state()
        self.assertEqual(len(self.bank._spike_history), 0)
        self.assertEqual(len(self.bank._voltage_history), 0)

    def test_graded_outputs_range(self):
        n = self.bank.n_total
        self.bank.step(np.random.rand(n))
        graded = self.bank.get_graded_outputs()
        self.assertEqual(graded.shape, (n,))
        self.assertTrue((graded >= 0.0).all())
        self.assertTrue((graded <= 1.0).all())

    def test_no_nan_after_many_steps(self):
        n = self.bank.n_total
        for _ in range(20):
            spk, mem = self.bank.step(np.random.rand(n) * 10)
        self.assertFalse(np.isnan(spk).any())
        self.assertFalse(np.isnan(mem).any())


if __name__ == "__main__":
    unittest.main()
