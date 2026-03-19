"""Tests for WormEnv."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from celegans.environment import WormEnv


class TestWormEnv(unittest.TestCase):

    def setUp(self):
        self.env = WormEnv(
            n_neurons=10, body_segments=4, physics_substeps=2,
            food_gradient_strength=1.0, num_motor_neurons=4, seed=42,
        )

    def test_observation_space_valid(self):
        obs, _ = self.env.reset(seed=42)
        for key, space in self.env.observation_space.spaces.items():
            self.assertTrue(
                space.contains(obs[key]),
                f"Observation '{key}' out of bounds: min={obs[key].min():.3f} max={obs[key].max():.3f}"
            )

    def test_reset_returns_correct_keys(self):
        obs, info = self.env.reset(seed=0)
        expected = {"neural_state", "body_position", "food_gradient", "spike_rates"}
        self.assertEqual(set(obs.keys()), expected)
        self.assertIsInstance(info, dict)

    def test_reset_randomizes_positions(self):
        obs1, _ = self.env.reset(seed=1)
        obs2, _ = self.env.reset(seed=2)
        self.assertFalse(np.allclose(obs1["body_position"], obs2["body_position"]))

    def test_step_returns_correct_types(self):
        self.env.reset(seed=0)
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_render_returns_rgb(self):
        self.env.reset(seed=0)
        frame = self.env.render()
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (self.env.num_motor_neurons,))

    def test_multiple_steps_no_crash(self):
        self.env.reset(seed=0)
        obs = None
        for _ in range(20):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break
        self.assertIsNotNone(obs)
        for key, space in self.env.observation_space.spaces.items():
            self.assertTrue(space.contains(obs[key]),
                            f"Obs '{key}' out of bounds after steps")

    def test_food_gradient_shape(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs["food_gradient"].shape, (2,))
        self.assertTrue((obs["food_gradient"] >= -1.0).all())
        self.assertTrue((obs["food_gradient"] <= 1.0).all())

    def test_update_neural_state(self):
        self.env.reset(seed=0)
        new_mem = np.ones(self.env.n_neurons, dtype=np.float32) * 0.5
        new_rates = np.ones(self.env.n_neurons, dtype=np.float32) * 0.3
        self.env.update_neural_state(new_mem, new_rates)
        obs = self.env._get_obs()
        self.assertTrue((obs["neural_state"] != 0).any())


if __name__ == "__main__":
    unittest.main()
