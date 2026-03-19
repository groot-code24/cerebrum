"""Tests for SimulationRunner and EpisodeResult."""
import sys, os, unittest, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from pathlib import Path
from conftest import make_mock_graph, make_mock_runner
from celegans.simulation import EpisodeResult


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.graph = make_mock_graph()
        self.runner = make_mock_runner(self.graph, self.tmp)

    def test_episode_runs_without_error(self):
        result = self.runner.run_episode(episode_seed=42)
        self.assertIsInstance(result, EpisodeResult)

    def test_episode_result_fields(self):
        result = self.runner.run_episode(episode_seed=7)
        self.assertIsInstance(result.spike_history, np.ndarray)
        self.assertEqual(result.spike_history.ndim, 2)
        self.assertEqual(result.spike_history.shape[1], self.graph.data.num_nodes)
        self.assertIsInstance(result.trajectory, np.ndarray)
        self.assertEqual(result.trajectory.ndim, 2)
        self.assertEqual(result.trajectory.shape[1], 2)
        self.assertIsInstance(result.food_reached, bool)
        self.assertGreaterEqual(result.total_displacement, 0.0)
        for key in ("sensory", "interneuron", "motor"):
            self.assertIn(key, result.mean_spike_rate_by_type)
        self.assertGreater(result.episode_steps, 0)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_reproducibility(self):
        r1 = self.runner.run_episode(episode_seed=42)
        self.runner.lif_bank.reset_state()
        r2 = self.runner.run_episode(episode_seed=42)
        np.testing.assert_array_equal(r1.spike_history, r2.spike_history)

    def test_trajectory_length(self):
        result = self.runner.run_episode()
        self.assertLessEqual(result.trajectory.shape[0], self.runner.sim_steps)

    def test_summary_serialisable(self):
        result = self.runner.run_episode()
        summary = result.summary()
        json.dumps(summary)  # must not raise

    def test_results_saved_to_disk(self):
        self.runner.run_episode()
        json_files = list(self.tmp.glob("*_summary.json"))
        self.assertGreaterEqual(len(json_files), 1)
        with open(json_files[0]) as f:
            data = json.load(f)
        self.assertIn("food_reached", data)


if __name__ == "__main__":
    unittest.main()
