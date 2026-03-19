"""Main simulation loop — connects GNN, LIF bank, and WormEnv."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from celegans.connectome import ConnectomeGraph
from celegans.environment import WormEnv
from celegans.gnn_model import ConnectomeGNN
from celegans.lif_neurons import LIFNeuronBank
from celegans.utils.io import atomic_write_json, validate_path
from celegans.utils.logging import get_logger
from celegans.utils.reproducibility import set_all_seeds

logger = get_logger(__name__)


@dataclass
class EpisodeResult:
    """Typed container for a single simulation episode result."""

    spike_history: np.ndarray          # [T, N]
    trajectory: np.ndarray             # [T, 2]
    food_reached: bool
    total_displacement: float
    mean_spike_rate_by_type: Dict[str, float]
    episode_steps: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "food_reached": self.food_reached,
            "total_displacement": round(self.total_displacement, 4),
            "episode_steps": self.episode_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "mean_spike_rate_by_type": {
                k: round(float(v), 6) for k, v in self.mean_spike_rate_by_type.items()
            },
            "spike_history_shape": list(self.spike_history.shape),
            "trajectory_shape": list(self.trajectory.shape),
        }


class SimulationRunner:
    """Runs the closed-loop C. elegans connectome emulation.

    Args:
        graph: Connectome graph (may be ablated).
        model: ConnectomeGNN instance.
        lif_bank: LIFNeuronBank instance.
        env: WormEnv instance.
        sim_steps: Max timesteps per episode.
        results_dir: Directory for saving episode outputs.
        seed: Random seed.
        project_root: Project root for path validation.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        model: ConnectomeGNN,
        lif_bank: LIFNeuronBank,
        env: WormEnv,
        sim_steps: int = 1000,
        results_dir: Optional[Path] = None,
        seed: int = 42,
        project_root: Optional[Path] = None,
    ) -> None:
        self.graph = graph
        self.model = model
        self.lif_bank = lif_bank
        self.env = env
        self.sim_steps = sim_steps
        self.seed = seed
        self._project_root = project_root or Path.cwd()

        if results_dir is not None:
            self.results_dir = validate_path(results_dir, self._project_root)
        else:
            self.results_dir = (self._project_root / "experiments" / "results").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(self, episode_seed: Optional[int] = None) -> EpisodeResult:
        """Run one complete simulation episode."""
        ep_seed = episode_seed if episode_seed is not None else self.seed
        set_all_seeds(ep_seed)
        t_start = time.perf_counter()

        obs, _ = self.env.reset(seed=ep_seed)
        self.lif_bank.reset_state()
        self.model.eval()

        trajectory: List[np.ndarray] = []
        food_reached = False

        for step in range(self.sim_steps):
            # 1. Sensory input
            gradient = obs["food_gradient"]
            n_sensory = len(self.graph.get_sensory_indices())
            sensory_input = np.zeros(max(n_sensory, 1), dtype=np.float32)
            if n_sensory >= 2:
                sensory_input[0] = float(gradient[0])
                sensory_input[1] = float(gradient[1])

            # 2. GNN forward pass → [N, 1]
            gnn_out = self.model(self.graph.data, sensory_input)

            # 3. LIF bank step
            spikes, mem_pots = self.lif_bank.step(gnn_out.ravel())

            # 4. Motor rates → action
            motor_rates = self.lif_bank.get_recent_spike_rates(window=10)
            motor_indices = self.graph.get_motor_indices()
            n_motor = self.env.num_motor_neurons
            action = np.zeros(n_motor, dtype=np.float32)
            for j, mi in enumerate(motor_indices[:n_motor]):
                action[j] = float(np.clip(motor_rates[mi], -1.0, 1.0))

            # 5. Env step
            obs, _reward, terminated, _truncated, _info = self.env.step(action)

            self.env.update_neural_state(mem_pots, motor_rates)
            trajectory.append(obs["body_position"][0].copy())

            if terminated:
                food_reached = True
                logger.info("Food reached at step %d", step + 1)
                break

        spike_hist = self.lif_bank.get_spike_history()
        traj_array = np.stack(trajectory, axis=0) if trajectory else np.zeros((1, 2))
        total_disp = float(np.linalg.norm(traj_array[-1] - traj_array[0]))
        mean_rates = _compute_mean_rates_by_type(spike_hist, self.graph)
        elapsed = time.perf_counter() - t_start

        result = EpisodeResult(
            spike_history=spike_hist,
            trajectory=traj_array,
            food_reached=food_reached,
            total_displacement=total_disp,
            mean_spike_rate_by_type=mean_rates,
            episode_steps=len(trajectory),
            elapsed_seconds=elapsed,
        )
        logger.info("Episode complete: %s", result.summary())
        self._save_episode(result)
        return result

    def _save_episode(self, result: EpisodeResult, label: str = "episode") -> None:
        ts = int(time.time())
        summary_path = self.results_dir / f"{label}_{ts}_summary.json"
        traj_path = self.results_dir / f"{label}_{ts}_trajectory.npy"
        atomic_write_json(summary_path, result.summary())
        fd, tmp = tempfile.mkstemp(dir=self.results_dir, suffix=".npy")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, result.trajectory)
            os.replace(tmp, traj_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _compute_mean_rates_by_type(
    spike_hist: np.ndarray, graph: ConnectomeGraph
) -> Dict[str, float]:
    if spike_hist.shape[0] == 0:
        return {"sensory": 0.0, "interneuron": 0.0, "motor": 0.0}
    mean_all = spike_hist.mean(axis=0)

    def _mean(indices: List[int]) -> float:
        if not indices:
            return 0.0
        return float(mean_all[np.array(indices, dtype=np.int64)].mean())

    return {
        "sensory": _mean(graph.get_sensory_indices()),
        "interneuron": _mean(graph.get_interneuron_indices()),
        "motor": _mean(graph.get_motor_indices()),
    }


def build_simulation_runner(
    graph: ConnectomeGraph,
    sim_steps: int = 1000,
    results_dir: Optional[Path] = None,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> SimulationRunner:
    """Convenience factory wiring all components from config."""
    from celegans.config import load_config
    cfg = load_config()

    sensory_idx = graph.get_sensory_indices()
    motor_idx = graph.get_motor_indices()
    interneuron_idx = graph.get_interneuron_indices()
    input_dim = graph.data.x.shape[1]

    model = ConnectomeGNN(
        input_dim=input_dim,
        hidden_dim=cfg.gnn_hidden_dim,
        num_layers=cfg.gnn_num_layers,
        dropout=cfg.gnn_dropout,
        aggr=cfg.gnn_aggr,
        sensory_indices=sensory_idx,
        motor_indices=motor_idx,
    )
    lif_bank = LIFNeuronBank(
        n_total=graph.data.num_nodes,
        sensory_indices=sensory_idx,
        interneuron_indices=interneuron_idx,
        motor_indices=motor_idx,
        tau_sensory=10.0,
        tau_interneuron=cfg.tau_mem,
        tau_motor=15.0,
        dt=cfg.dt,
        threshold=cfg.threshold,
        reset_potential=cfg.reset_potential,
    )
    n_motor = max(len(motor_idx), 1)
    dorsal = motor_idx[:n_motor // 2]
    ventral = motor_idx[n_motor // 2:]
    env = WormEnv(
        n_neurons=graph.data.num_nodes,
        body_segments=cfg.body_segments,
        physics_substeps=cfg.physics_substeps,
        food_gradient_strength=cfg.food_gradient_strength,
        dorsal_motor_indices=dorsal,
        ventral_motor_indices=ventral,
        num_motor_neurons=n_motor,
        seed=seed,
    )
    root = project_root or Path.cwd()
    res_dir = results_dir or (root / cfg.results_dir)
    return SimulationRunner(
        graph=graph, model=model, lif_bank=lif_bank, env=env,
        sim_steps=sim_steps, results_dir=res_dir, seed=seed, project_root=root,
    )
