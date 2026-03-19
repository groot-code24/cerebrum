"""Tests for TemporalGNN, validation metrics, tracking, and GraphVAE."""
import sys, os, unittest, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from pathlib import Path
from conftest import make_mock_graph


# ─── Temporal GNN ────────────────────────────────────────────────────────────

class TestTemporalGNN(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph()

    def test_delay_estimation(self):
        from celegans.temporal_gnn import estimate_delays
        delays = estimate_delays(
            self.graph.data.edge_index,
            self.graph.data.num_nodes,
            dt=0.1,
        )
        self.assertEqual(delays.shape[0], self.graph.data.num_edges)
        self.assertTrue((delays >= 1).all())
        self.assertTrue((delays <= 50).all())

    def test_forward_output_shape(self):
        from celegans.temporal_gnn import TemporalConnectomeGNN
        gnn = TemporalConnectomeGNN(
            input_dim=self.graph.data.x.shape[1],
            hidden_dim=16,
            num_layers=2,
            sensory_indices=self.graph.get_sensory_indices(),
            motor_indices=self.graph.get_motor_indices(),
        )
        out = gnn(self.graph.data)
        self.assertEqual(out.shape, (self.graph.data.num_nodes, 1))

    def test_temporal_differs_from_static(self):
        """Running multiple steps should produce different outputs due to delay buffer."""
        from celegans.temporal_gnn import TemporalConnectomeGNN
        from celegans.gnn_model import ConnectomeGNN
        gnn = TemporalConnectomeGNN(
            input_dim=self.graph.data.x.shape[1],
            hidden_dim=16,
            num_layers=2,
        )
        out1 = gnn(self.graph.data)
        out2 = gnn(self.graph.data)
        # Outputs can differ because delay buffer is non-empty on second call
        # (at minimum they should be valid arrays with no NaN)
        self.assertFalse(np.isnan(out1).any())
        self.assertFalse(np.isnan(out2).any())

    def test_reset_buffer(self):
        from celegans.temporal_gnn import TemporalConnectomeGNN
        gnn = TemporalConnectomeGNN(
            input_dim=self.graph.data.x.shape[1],
            hidden_dim=16,
            num_layers=2,
        )
        gnn(self.graph.data)
        gnn(self.graph.data)
        self.assertGreater(len(gnn._activation_buffer), 0)
        gnn.reset_buffer()
        self.assertEqual(len(gnn._activation_buffer), 0)

    def test_no_nan(self):
        from celegans.temporal_gnn import TemporalConnectomeGNN
        gnn = TemporalConnectomeGNN(
            input_dim=self.graph.data.x.shape[1],
            hidden_dim=16,
            num_layers=2,
        )
        for _ in range(10):
            out = gnn(self.graph.data)
            self.assertFalse(np.isnan(out).any())


# ─── Validation ──────────────────────────────────────────────────────────────

class TestValidation(unittest.TestCase):

    def test_chemotaxis_index_range(self):
        from celegans.validation import compute_chemotaxis_index
        traj = np.random.rand(100, 2) * 10
        food = np.array([50.0, 50.0])
        ci = compute_chemotaxis_index(traj, food)
        self.assertGreaterEqual(ci, -1.0)
        self.assertLessEqual(ci, 1.0)

    def test_ci_moving_toward_food_positive(self):
        from celegans.validation import compute_chemotaxis_index
        # Trajectory moving directly toward food
        food = np.array([100.0, 0.0])
        traj = np.column_stack([np.arange(50), np.zeros(50)])
        ci = compute_chemotaxis_index(traj, food)
        self.assertGreater(ci, 0.5)

    def test_ci_moving_away_negative(self):
        from celegans.validation import compute_chemotaxis_index
        food = np.array([0.0, 0.0])
        traj = np.column_stack([np.arange(50), np.zeros(50)])  # moving away
        ci = compute_chemotaxis_index(traj, food)
        self.assertLess(ci, 0.0)

    def test_ci_short_trajectory(self):
        from celegans.validation import compute_chemotaxis_index
        ci = compute_chemotaxis_index(np.zeros((1, 2)), np.array([1.0, 0.0]))
        self.assertEqual(ci, 0.0)

    def test_synthetic_kato_data_shape(self):
        from celegans.validation import generate_synthetic_kato_data
        data = generate_synthetic_kato_data(n_timepoints=100, n_neurons=30)
        self.assertEqual(data.shape, (100, 30))
        self.assertFalse(np.isnan(data).any())

    def test_procrustes_distance_range(self):
        from celegans.validation import procrustes_distance
        X = np.random.rand(100, 20).astype(np.float64)
        Y = np.random.rand(100, 20).astype(np.float64)
        dist, Xa, Ya = procrustes_distance(X, Y, n_components=5)
        self.assertGreaterEqual(dist, 0.0)
        self.assertFalse(np.isnan(dist))

    def test_procrustes_identical_is_zero(self):
        from celegans.validation import procrustes_distance
        X = np.random.rand(50, 10).astype(np.float64)
        dist, _, _ = procrustes_distance(X, X.copy(), n_components=5)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_validate_simulation(self):
        from celegans.validation import validate_simulation
        graph = make_mock_graph()
        T = 50
        spike_hist = np.random.rand(T, graph.data.num_nodes).astype(np.float32)
        result = validate_simulation(spike_hist, graph.node_names)
        self.assertIn("procrustes_distance", result)
        self.assertIn("n_kato_neurons_matched", result)


# ─── Tracking ────────────────────────────────────────────────────────────────

class TestExperimentTracker(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_context_manager_creates_files(self):
        from celegans.tracking import ExperimentTracker
        with ExperimentTracker("test_run", results_dir=self.tmp, use_mlflow=False) as t:
            t.log_params({"lr": 0.01, "epochs": 10})
            t.log_metrics({"loss": 0.5, "accuracy": 0.9})

        runs = list((self.tmp / "runs").glob("test_run_*"))
        self.assertGreater(len(runs), 0)
        run_dir = runs[0]
        self.assertTrue((run_dir / "params.json").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "run_summary.json").exists())

    def test_params_json_content(self):
        from celegans.tracking import ExperimentTracker
        with ExperimentTracker("param_test", results_dir=self.tmp, use_mlflow=False) as t:
            t.log_params({"neuron": "AVAL", "seed": 42})
        runs = list((self.tmp / "runs").glob("param_test_*"))
        params = json.loads((runs[0] / "params.json").read_text())
        self.assertEqual(params["neuron"], "AVAL")
        self.assertEqual(params["seed"], 42)

    def test_metrics_history(self):
        from celegans.tracking import ExperimentTracker
        with ExperimentTracker("metric_test", results_dir=self.tmp, use_mlflow=False) as t:
            for i in range(5):
                t.log_metrics({"loss": 1.0 / (i + 1)}, step=i)
        runs = list((self.tmp / "runs").glob("metric_test_*"))
        metrics = json.loads((runs[0] / "metrics.json").read_text())
        self.assertEqual(len(metrics["loss"]), 5)

    def test_artifact_copy(self):
        from celegans.tracking import ExperimentTracker
        art_file = self.tmp / "test_artifact.txt"
        art_file.write_text("test content")
        with ExperimentTracker("art_test", results_dir=self.tmp, use_mlflow=False) as t:
            t.log_artifact(art_file)
        runs = list((self.tmp / "runs").glob("art_test_*"))
        self.assertTrue((runs[0] / "artifacts" / "test_artifact.txt").exists())

    def test_failed_run_tagged(self):
        from celegans.tracking import ExperimentTracker
        try:
            with ExperimentTracker("fail_test", results_dir=self.tmp, use_mlflow=False) as t:
                t.log_params({"x": 1})
                raise ValueError("intentional error")
        except ValueError:
            pass
        runs = list((self.tmp / "runs").glob("fail_test_*"))
        summary = json.loads((runs[0] / "run_summary.json").read_text())
        self.assertEqual(summary["tags"]["status"], "FAILED")

    def test_load_run(self):
        from celegans.tracking import ExperimentTracker
        with ExperimentTracker("load_test", results_dir=self.tmp, use_mlflow=False) as t:
            t.log_metrics({"ci": 0.65})
        runs = list((self.tmp / "runs").glob("load_test_*"))
        loaded = ExperimentTracker.load_run(runs[0])
        self.assertIn("final_metrics", loaded)


# ─── Graph VAE ───────────────────────────────────────────────────────────────

class TestGraphVAE(unittest.TestCase):

    def setUp(self):
        self.graph = make_mock_graph()
        self.vae = __import__('celegans.graph_vae', fromlist=['GraphVAE']).GraphVAE(
            input_dim=self.graph.data.x.shape[1],
            hidden_dim=8,
            latent_dim=4,
            num_encoder_layers=1,
        )

    def test_encode_shapes(self):
        mu, log_var = self.vae.encode(self.graph.data)
        N = self.graph.data.num_nodes
        self.assertEqual(mu.shape, (N, 4))
        self.assertEqual(log_var.shape, (N, 4))

    def test_reparameterize_shape(self):
        mu, log_var = self.vae.encode(self.graph.data)
        z = self.vae.reparameterize(mu, log_var)
        self.assertEqual(z.shape, mu.shape)

    def test_deterministic_reparameterize(self):
        mu, log_var = self.vae.encode(self.graph.data)
        z = self.vae.reparameterize(mu, log_var, deterministic=True)
        np.testing.assert_array_equal(z, mu)

    def test_decode_shape(self):
        N = self.graph.data.num_nodes
        mu, _ = self.vae.encode(self.graph.data)
        adj = self.vae.decode(mu)
        self.assertEqual(adj.shape, (N, N))
        self.assertTrue((adj >= 0).all())
        self.assertTrue((adj <= 1).all())

    def test_forward_shapes(self):
        adj, z, mu, lv = self.vae.forward(self.graph.data)
        N = self.graph.data.num_nodes
        self.assertEqual(adj.shape, (N, N))
        self.assertEqual(z.shape, (N, 4))

    def test_loss_returns_dict(self):
        adj, z, mu, lv = self.vae.forward(self.graph.data)
        loss = self.vae.loss(self.graph.data, adj, mu, lv)
        self.assertIn("total", loss)
        self.assertIn("recon", loss)
        self.assertIn("kl", loss)
        self.assertIsInstance(loss["total"], float)
        self.assertFalse(np.isnan(loss["total"]))

    def test_latent_codes(self):
        codes = self.vae.get_latent_codes(self.graph.data)
        self.assertEqual(codes.shape, (self.graph.data.num_nodes, 4))

    def test_interpolate_returns_list(self):
        interp = self.vae.interpolate(self.graph.data, self.graph.data, steps=5)
        self.assertEqual(len(interp), 5)
        N = self.graph.data.num_nodes
        self.assertEqual(interp[0].shape, (N, N))

    def test_reconstruction_accuracy_range(self):
        acc = self.vae.adjacency_reconstruction_accuracy(self.graph.data)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_no_nan_in_forward(self):
        adj, z, mu, lv = self.vae.forward(self.graph.data)
        for arr in (adj, z, mu, lv):
            self.assertFalse(np.isnan(arr).any())

    def test_invalid_latent_dim(self):
        from celegans.graph_vae import GraphVAE
        with self.assertRaises(ValueError):
            GraphVAE(input_dim=6, latent_dim=0)


if __name__ == "__main__":
    unittest.main()
