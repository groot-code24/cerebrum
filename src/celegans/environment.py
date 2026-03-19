"""2D worm locomotion and chemotaxis environment.

Pure numpy Verlet-physics body. Gymnasium-compatible interface when
gymnasium is installed; otherwise provides the same API with plain Python.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from celegans.utils.logging import get_logger

logger = get_logger(__name__)

# Physics constants
_SPRING_K = 30.0
_DAMPING = 0.98
_SEGMENT_REST_LEN = 2.0
_MUSCLE_FORCE = 5.0
_FOOD_REACH_DIST = 5.0


# ---------------------------------------------------------------------------
# Minimal space classes (mirrors gymnasium.spaces API)
# ---------------------------------------------------------------------------

class _Box:
    """Minimal Box space for type checking / contains()."""
    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple,
        dtype: type = np.float32,
    ) -> None:
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def contains(self, x: np.ndarray) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return bool(
            arr.shape == self.shape
            and (arr >= self.low).all()
            and (arr <= self.high).all()
        )

    def sample(self) -> np.ndarray:
        rng = np.random.default_rng()
        return (rng.random(self.shape) * (self.high - self.low) + self.low).astype(self.dtype)


class _DictSpace:
    def __init__(self, spaces: Dict[str, _Box]) -> None:
        self.spaces = spaces


# Try gymnasium first
try:
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class WormEnv:
    """2D worm locomotion / chemotaxis environment.

    Observation keys:
        neural_state    : [n_neurons]       membrane potentials
        body_position   : [body_segments,2] segment (x,y)
        food_gradient   : [2]               gradient direction * magnitude
        spike_rates     : [n_neurons]       recent mean spike rate

    Action: float array [num_motor_neurons] in [-1, 1].
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        n_neurons: int = 302,
        body_segments: int = 10,
        physics_substeps: int = 5,
        food_gradient_strength: float = 1.0,
        dorsal_motor_indices: Optional[list] = None,
        ventral_motor_indices: Optional[list] = None,
        num_motor_neurons: int = 50,
        render_size: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        self.n_neurons = n_neurons
        self.body_segments = body_segments
        self.physics_substeps = physics_substeps
        self.food_gradient_strength = food_gradient_strength
        self.num_motor_neurons = num_motor_neurons
        self.render_size = render_size

        self.dorsal_motor_indices: list = dorsal_motor_indices or list(range(0, max(1, num_motor_neurons // 2)))
        self.ventral_motor_indices: list = ventral_motor_indices or list(range(max(1, num_motor_neurons // 2), num_motor_neurons))

        self.observation_space = _DictSpace({
            "neural_state":  _Box(-5.0,   5.0,   (n_neurons,)),
            "body_position": _Box(-200.0, 200.0, (body_segments, 2)),
            "food_gradient": _Box(-1.0,   1.0,   (2,)),
            "spike_rates":   _Box(0.0,    1.0,   (n_neurons,)),
        })
        self.action_space = _Box(-1.0, 1.0, (num_motor_neurons,))

        self._rng = np.random.default_rng(seed)
        self._pos = np.zeros((body_segments, 2), dtype=np.float64)
        self._vel = np.zeros((body_segments, 2), dtype=np.float64)
        self._food_pos = np.zeros(2, dtype=np.float64)
        self._neural_state = np.zeros(n_neurons, dtype=np.float32)
        self._spike_rates = np.zeros(n_neurons, dtype=np.float32)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start_x = float(self._rng.uniform(-50, 50))
        start_y = float(self._rng.uniform(-50, 50))
        angle = float(self._rng.uniform(0, 2 * np.pi))

        for i in range(self.body_segments):
            self._pos[i, 0] = start_x - i * _SEGMENT_REST_LEN * np.cos(angle)
            self._pos[i, 1] = start_y - i * _SEGMENT_REST_LEN * np.sin(angle)
        self._vel[:] = 0.0

        food_angle = float(self._rng.uniform(0, 2 * np.pi))
        food_dist = float(self._rng.uniform(60, 100))
        self._food_pos[0] = start_x + food_dist * np.cos(food_angle)
        self._food_pos[1] = start_y + food_dist * np.sin(food_angle)

        self._neural_state = self._rng.uniform(-0.1, 0.1, self.n_neurons).astype(np.float32)
        self._spike_rates = np.zeros(self.n_neurons, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        prev_head = self._pos[0].copy()

        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        n_d = len(self.dorsal_motor_indices)
        dorsal_force = float(action[:n_d].mean()) if n_d else 0.0
        ventral_force = float(action[n_d:].mean()) if len(action) > n_d else 0.0

        for _ in range(self.physics_substeps):
            self._verlet_step(dorsal_force, ventral_force)

        head = self._pos[0]
        food_vec = self._food_pos - head
        food_dist = float(np.linalg.norm(food_vec))
        food_dir = food_vec / food_dist if food_dist > 1e-6 else np.zeros(2)
        concentration = self.food_gradient_strength * np.exp(-food_dist ** 2 / (2 * 50 ** 2))

        self._neural_state = (self._neural_state * 0.9).astype(np.float32)
        self._neural_state[:2] = (food_dir * concentration).astype(np.float32)
        self._step_count += 1
        food_reached = food_dist < _FOOD_REACH_DIST

        displacement = float(np.dot(self._pos[0] - prev_head, food_dir))
        reward = 1.0 if displacement > 0 else -0.1
        if food_reached:
            reward += 10.0

        return (
            self._get_obs(),
            reward,
            food_reached,
            False,
            {"food_distance": food_dist, "step": self._step_count},
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        H = W = self.render_size
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")

        xs = np.linspace(-100, 100, 50)
        ys = np.linspace(-100, 100, 50)
        XX, YY = np.meshgrid(xs, ys)
        dist_sq = (XX - self._food_pos[0]) ** 2 + (YY - self._food_pos[1]) ** 2
        Z = self.food_gradient_strength * np.exp(-dist_sq / (2 * 50 ** 2))
        ax.contourf(XX, YY, Z, levels=10, cmap="YlOrRd", alpha=0.4)

        xw, yw = self._pos[:, 0], self._pos[:, 1]
        ax.plot(xw, yw, "o-", color="#00ff88", linewidth=2, markersize=6)
        ax.plot(xw[0], yw[0], "o", color="#ffffff", markersize=9, label="Head")
        ax.plot(self._food_pos[0], self._food_pos[1], "*", color="#ffdd00",
                markersize=15, label="Food")
        ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(H, W, 4)[:, :, :3]
        plt.close(fig)
        return buf

    def close(self) -> None:
        plt.close("all")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verlet_step(self, dorsal_force: float, ventral_force: float) -> None:
        dt_sub = 1.0 / max(self.physics_substeps, 1)
        forces = np.zeros_like(self._pos)

        for i in range(self.body_segments - 1):
            diff = self._pos[i + 1] - self._pos[i]
            dist = float(np.linalg.norm(diff))
            if dist < 1e-9:
                continue
            direction = diff / dist
            stretch = dist - _SEGMENT_REST_LEN
            f_spring = _SPRING_K * stretch * direction
            forces[i] += f_spring
            forces[i + 1] -= f_spring

        for i in range(1, self.body_segments):
            seg_dir = self._pos[i] - self._pos[i - 1]
            seg_len = float(np.linalg.norm(seg_dir))
            if seg_len < 1e-9:
                continue
            perp = np.array([-seg_dir[1], seg_dir[0]]) / seg_len
            phase = i * np.pi / self.body_segments
            muscle = dorsal_force * np.cos(phase) + ventral_force * np.sin(phase)
            forces[i] += perp * muscle * _MUSCLE_FORCE

        self._vel += forces * dt_sub
        self._vel *= _DAMPING
        self._pos += self._vel * dt_sub
        # Guard against physics blow-up
        self._pos = np.nan_to_num(self._pos, nan=0.0, posinf=100.0, neginf=-100.0)
        self._vel = np.nan_to_num(self._vel, nan=0.0, posinf=1.0,   neginf=-1.0)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        head = self._pos[0]
        food_vec = self._food_pos - head
        food_dist = float(np.linalg.norm(food_vec))
        food_dir = (food_vec / food_dist).astype(np.float32) if food_dist > 1e-6 else np.zeros(2, dtype=np.float32)
        concentration = float(self.food_gradient_strength * np.exp(-food_dist ** 2 / (2 * 50 ** 2)))
        gradient = np.clip(food_dir * concentration, -1.0, 1.0).astype(np.float32)
        return {
            "neural_state":  np.clip(self._neural_state, -5.0, 5.0).astype(np.float32),
            "body_position": np.clip(self._pos, -200.0, 200.0).astype(np.float32),
            "food_gradient": gradient,
            "spike_rates":   np.clip(self._spike_rates, 0.0, 1.0).astype(np.float32),
        }

    def update_neural_state(
        self, membrane_potentials: np.ndarray, spike_rates: np.ndarray
    ) -> None:
        self._neural_state = np.clip(
            np.asarray(membrane_potentials, dtype=np.float32), -5.0, 5.0
        )
        self._spike_rates = np.clip(
            np.asarray(spike_rates, dtype=np.float32), 0.0, 1.0
        )
