"""Leaky Integrate-and-Fire neuron dynamics — pure numpy implementation.

When snnTorch is available it is used; otherwise numpy LIF math is applied.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from celegans.utils.logging import get_logger

logger = get_logger(__name__)


def _beta_from_tau(tau_mem: float, dt: float) -> float:
    return math.exp(-dt / tau_mem)


class LIFLayer:
    """Single-group LIF neuron layer (numpy).

    Args:
        num_neurons: Population size.
        tau_mem: Membrane time constant (ms).
        dt: Simulation timestep (ms).
        threshold: Spike threshold.
        reset_potential: Post-spike reset value.
    """

    def __init__(
        self,
        num_neurons: int,
        tau_mem: float = 20.0,
        dt: float = 0.1,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
    ) -> None:
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.reset_potential = reset_potential
        self._beta = float(_beta_from_tau(tau_mem, dt))
        self.mem = np.full(num_neurons, reset_potential, dtype=np.float64)

    def step(self, input_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one timestep.

        Returns:
            (spikes [N], membrane_potential [N])
        """
        cur = np.asarray(input_current, dtype=np.float64).ravel()[:self.num_neurons]
        if len(cur) < self.num_neurons:
            cur = np.pad(cur, (0, self.num_neurons - len(cur)))

        new_mem = self._beta * self.mem + cur
        spikes = (new_mem >= self.threshold).astype(np.float32)
        new_mem -= spikes * (new_mem - self.reset_potential)
        self.mem = new_mem
        return spikes, new_mem.astype(np.float32)

    def reset_state(self) -> None:
        self.mem = np.full(self.num_neurons, self.reset_potential, dtype=np.float64)


class LIFNeuronBank:
    """Full 302-neuron LIF bank with per-group time constants.

    Args:
        n_total: Total neuron count.
        sensory_indices: Indices of sensory neurons.
        interneuron_indices: Indices of interneurons.
        motor_indices: Indices of motor neurons.
        tau_sensory: Sensory membrane τ (ms).
        tau_interneuron: Interneuron membrane τ (ms).
        tau_motor: Motor neuron membrane τ (ms).
        dt: Simulation timestep (ms).
        threshold: Spike threshold.
        reset_potential: Post-spike reset.
        use_learned_transform: Unused in numpy mode.
    """

    def __init__(
        self,
        n_total: int = 302,
        sensory_indices: Optional[List[int]] = None,
        interneuron_indices: Optional[List[int]] = None,
        motor_indices: Optional[List[int]] = None,
        tau_sensory: float = 10.0,
        tau_interneuron: float = 20.0,
        tau_motor: float = 15.0,
        dt: float = 0.1,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
        use_learned_transform: bool = False,
    ) -> None:
        self.n_total = n_total
        self.sensory_indices: List[int] = sensory_indices or []
        self.interneuron_indices: List[int] = interneuron_indices or []
        self.motor_indices: List[int] = motor_indices or []

        n_s = max(len(self.sensory_indices), 1)
        n_i = max(len(self.interneuron_indices), 1)
        n_m = max(len(self.motor_indices), 1)

        self.sensory_lif = LIFLayer(n_s, tau_sensory, dt, threshold, reset_potential)
        self.interneuron_lif = LIFLayer(n_i, tau_interneuron, dt, threshold, reset_potential)
        self.motor_lif = LIFLayer(n_m, tau_motor, dt, threshold, reset_potential)

        self._spike_history: List[np.ndarray] = []

    def step(
        self, gnn_activations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance all groups one timestep.

        Args:
            gnn_activations: Shape [n_total] or [n_total, 1].

        Returns:
            (spikes [n_total], membrane_potentials [n_total])
        """
        act = np.asarray(gnn_activations, dtype=np.float64).ravel()
        if len(act) > self.n_total:
            act = act[:self.n_total]
        elif len(act) < self.n_total:
            act = np.pad(act, (0, self.n_total - len(act)))

        spikes = np.zeros(self.n_total, dtype=np.float32)
        mems = np.zeros(self.n_total, dtype=np.float32)

        for indices, lif in [
            (self.sensory_indices, self.sensory_lif),
            (self.interneuron_indices, self.interneuron_lif),
            (self.motor_indices, self.motor_lif),
        ]:
            if not indices:
                continue
            idx = np.array(indices, dtype=np.int64)
            group_act = act[idx]
            spk, mem = lif.step(group_act)
            n_g = min(len(idx), len(spk))
            spikes[idx[:n_g]] = spk[:n_g]
            mems[idx[:n_g]] = mem[:n_g]

        self._spike_history.append(spikes.copy())
        return spikes, mems

    def reset_state(self) -> None:
        """Clear spike history and reset all membrane potentials."""
        self._spike_history = []
        self.sensory_lif.reset_state()
        self.interneuron_lif.reset_state()
        self.motor_lif.reset_state()

    def get_spike_history(self) -> np.ndarray:
        """Return [T, n_total] spike history array."""
        if not self._spike_history:
            return np.zeros((0, self.n_total), dtype=np.float32)
        return np.stack(self._spike_history, axis=0)

    def get_recent_spike_rates(self, window: int = 10) -> np.ndarray:
        """Mean spike rate over the last *window* timesteps. Shape [n_total]."""
        hist = self.get_spike_history()
        if hist.shape[0] == 0:
            return np.zeros(self.n_total, dtype=np.float32)
        return hist[-window:].mean(axis=0)
