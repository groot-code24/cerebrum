"""Tests for AblationExperiment."""
import sys, os, unittest, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from pathlib import Path
from conftest import make_mock_graph, make_mock_runner
from celegans.ablation import AblationExperiment, AblationResult


class TestAblation(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.graph = make_mock_graph()
        self.runner = make_mock_runner(self.graph, self.tmp)
        self.exp = AblationExperiment(
            base_runner=self.runner,
            ablation_fractions=[0.0, 0.5],
            ablation_seeds=[42],
            specific_neurons=["AVAL", "AVBL"],
            results_dir=self.tmp,
            project_root=self.tmp,
            episode_seed=42,
        )

    def test_specific_ablation_returns_result(self):
        result = self.exp.run_specific_ablation(["AVAL"])
        self.assertIsInstance(result, AblationResult)
        self.assertIsInstance(result.locomotion_score, float)
        self.assertIsInstance(result.chemotaxis_score, float)
        self.assertIsInstance(result.behavioral_degradation_pct, float)

    def test_specific_ablation_score_reasonable(self):
        neuron = self.runner.graph.node_names[0]
        result = self.exp.run_specific_ablation([neuron])
        self.assertLessEqual(result.locomotion_score, 2.0)

    def test_random_ablation_degrades_gracefully(self):
        r0  = self.exp.run_random_ablation(0.0, seed=42)
        r50 = self.exp.run_random_ablation(0.5, seed=42)
        self.assertLessEqual(r50.locomotion_score, r0.locomotion_score + 0.5)

    def test_results_saved_to_disk(self):
        self.exp.run_full_ablation_suite()
        json_path = self.tmp / "ablation_all_results.json"
        self.assertTrue(json_path.exists())
        with open(json_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_markdown_table_saved(self):
        self.exp.run_full_ablation_suite()
        md_path = self.tmp / "ablation_table.md"
        self.assertTrue(md_path.exists())
        content = md_path.read_text()
        self.assertIn("Locomotion Score", content)

    def test_result_to_dict_serialisable(self):
        r = AblationResult(
            ablated_neurons=["AVAL"], ablation_fraction=0.0, ablation_seed=-1,
            locomotion_score=0.75, chemotaxis_score=0.60,
            mean_spike_rate_change=-0.12, behavioral_degradation_pct=32.5,
            food_reached=False,
        )
        json.dumps(r.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
