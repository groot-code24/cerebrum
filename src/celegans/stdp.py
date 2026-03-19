"""Spike-Timing Dependent Plasticity (STDP) for the C. elegans connectome.

Implements three STDP variants:
  1. Classical asymmetric STDP  — for chemical (directed) synapses
  2. Symmetric STDP             — for gap junctions (bidirectional)
  3. Triplet STDP               — for more realistic synaptic dynamics
                                  (Pfister & Gerstner 2006)

Usage
-----
Create a :class:`STDPLearner` and call :meth:`update` after each simulation
timestep with the current spike vector. The learner maintains eligibility
traces and directly modifies ``edge_attr`` on a :class:`GraphData` object.

References
----------
Bi G, Poo M (1998). J Neurosci 18(24):10464–10472.
Pfister JP, Gerstner W (2006). J Neurosci 26(38):9673–9682.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from celegans.connectome import GraphData
from celegans.utils.logging import get_logger

logger = get_logger(__name__)


class STDPLearner:
    """Online STDP weight update for a connectome graph.

    After each simulation step, call :meth:`update` with the current spike
    vector. The learner maintains per-neuron pre- and post-synaptic traces
    and updates ``edge_attr`` (synapse weights) accordingly.

    Parameters
    ----------
    edge_index : np.ndarray
        Shape ``[2, E]`` — source and target indices.
    edge_attr : np.ndarray
        Shape ``[E, 1]`` — synapse weights (modified in-place).
    n_neurons : int
        Total neuron count.
    A_plus : float
        LTP magnitude (causal: pre before post).
    A_minus : float
        LTD magnitude (anti-causal: post before pre).
    tau_plus : float
        LTP time constant (ms).
    tau_minus : float
        LTD time constant (ms).
    w_min / w_max : float
        Hard weight bounds.
    dt : float
        Simulation timestep (ms).
    symmetric : bool
        If ``True``, use symmetric STDP suitable for gap junctions.
    learning_rate : float
        Global scaling factor applied to all weight changes.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        n_neurons: int,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 0.1,
        symmetric: bool = False,
        learning_rate: float = 1.0,
    ) -> None:
        self.edge_index = edge_index
        self.edge_attr = edge_attr  # reference — modified in-place
        self.n_neurons = n_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt
        self.symmetric = symmetric
        self.learning_rate = learning_rate

        # Decay factors per timestep
        self.decay_plus = math.exp(-dt / tau_plus) if tau_plus > 0 else 0.0
        self.decay_minus = math.exp(-dt / tau_minus) if tau_minus > 0 else 0.0

        # Eligibility traces
        self.x_pre = np.zeros(n_neurons, dtype=np.float64)   # pre-synaptic trace
        self.x_post = np.zeros(n_neurons, dtype=np.float64)  # post-synaptic trace

        self._total_updates = 0
        logger.debug(
            "STDPLearner: n_neurons=%d  edges=%d  A+/A-=%.4f/%.4f  symmetric=%s",
            n_neurons, edge_index.shape[1], A_plus, A_minus, symmetric,
        )

    def update(self, spikes: np.ndarray) -> np.ndarray:
        """Apply one STDP step given the current spike vector.

        Parameters
        ----------
        spikes : np.ndarray[float32, N]
            Binary spike array (0 or 1).

        Returns
        -------
        delta_w : np.ndarray[float32, E]
            Weight changes applied this step.
        """
        s = np.asarray(spikes, dtype=np.float64).ravel()
        fired = s > 0.5

        src = self.edge_index[0]   # shape [E]
        tgt = self.edge_index[1]   # shape [E]

        # --- LTP: pre fires → potentiate synapses where post trace is high
        delta_w = np.zeros(self.edge_attr.shape[0], dtype=np.float64)

        pre_fired_edges = fired[src]
        post_fired_edges = fired[tgt]

        # LTP: pre fires now, post fired recently (x_post[tgt] > 0)
        delta_w += pre_fired_edges * self.A_plus * self.x_post[tgt]

        # LTD: post fires now, pre fired recently (x_pre[src] > 0)
        if not self.symmetric:
            delta_w -= post_fired_edges * self.A_minus * self.x_pre[src]
        else:
            # Symmetric: bidirectional potentiation (gap junctions)
            delta_w += post_fired_edges * self.A_plus * self.x_pre[src]

        # Decay and update traces
        self.x_pre *= self.decay_plus
        self.x_post *= self.decay_minus
        self.x_pre[fired] += 1.0
        self.x_post[fired] += 1.0

        # Apply weight changes
        delta_w *= self.learning_rate
        new_w = np.clip(self.edge_attr[:, 0] + delta_w, self.w_min, self.w_max)
        self.edge_attr[:, 0] = new_w

        self._total_updates += 1
        return delta_w.astype(np.float32)

    def reset_traces(self) -> None:
        """Reset eligibility traces (call at episode start)."""
        self.x_pre[:] = 0.0
        self.x_post[:] = 0.0

    def weight_statistics(self) -> dict:
        """Summary statistics of current synapse weight distribution."""
        w = self.edge_attr[:, 0]
        return {
            "mean": float(w.mean()),
            "std": float(w.std()),
            "min": float(w.min()),
            "max": float(w.max()),
            "fraction_at_min": float((w <= self.w_min + 1e-6).mean()),
            "fraction_at_max": float((w >= self.w_max - 1e-6).mean()),
        }


# Need math for decay computation
import math


class TripletSTDP:
    """Triplet STDP rule (Pfister & Gerstner 2006).

    Extends pair-based STDP with additional slow traces that capture
    triplet spike interactions. More accurately fits experimental BCM curves.

    Parameters
    ----------
    A2_plus / A2_minus : float
        Pair interaction amplitudes.
    A3_plus / A3_minus : float
        Triplet interaction amplitudes.
    tau_x1 / tau_x2 : float
        Pre-synaptic fast/slow trace time constants (ms).
    tau_y1 / tau_y2 : float
        Post-synaptic fast/slow trace time constants (ms).
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        n_neurons: int,
        A2_plus: float = 0.005,
        A2_minus: float = 0.007,
        A3_plus: float = 0.006,
        A3_minus: float = 0.003,
        tau_x1: float = 16.8,
        tau_x2: float = 101.0,
        tau_y1: float = 33.7,
        tau_y2: float = 125.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 0.1,
        learning_rate: float = 1.0,
    ) -> None:
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.n = n_neurons
        self.A2p = A2_plus
        self.A2m = A2_minus
        self.A3p = A3_plus
        self.A3m = A3_minus
        self.w_min = w_min
        self.w_max = w_max
        self.lr = learning_rate

        self.d_x1 = math.exp(-dt / tau_x1)
        self.d_x2 = math.exp(-dt / tau_x2)
        self.d_y1 = math.exp(-dt / tau_y1)
        self.d_y2 = math.exp(-dt / tau_y2)

        self.x1 = np.zeros(n_neurons, dtype=np.float64)
        self.x2 = np.zeros(n_neurons, dtype=np.float64)
        self.y1 = np.zeros(n_neurons, dtype=np.float64)
        self.y2 = np.zeros(n_neurons, dtype=np.float64)

    def update(self, spikes: np.ndarray) -> np.ndarray:
        s = np.asarray(spikes, dtype=np.float64).ravel()
        fired = s > 0.5

        src = self.edge_index[0]
        tgt = self.edge_index[1]

        pre_fired = fired[src]
        post_fired = fired[tgt]

        # LTP: post fires — use fast pre trace + slow post trace
        dw_ltp = post_fired * (self.A2p * self.x1[src] + self.A3p * self.y2[tgt])

        # LTD: pre fires — use fast post trace + slow pre trace
        dw_ltd = pre_fired * (self.A2m * self.y1[tgt] + self.A3m * self.x2[src])

        delta_w = (dw_ltp - dw_ltd) * self.lr

        # Decay
        self.x1 *= self.d_x1
        self.x2 *= self.d_x2
        self.y1 *= self.d_y1
        self.y2 *= self.d_y2

        # Update traces for fired neurons
        self.x1[fired] += 1.0
        self.x2[fired] += 1.0
        self.y1[fired] += 1.0
        self.y2[fired] += 1.0

        new_w = np.clip(self.edge_attr[:, 0] + delta_w, self.w_min, self.w_max)
        self.edge_attr[:, 0] = new_w
        return delta_w.astype(np.float32)

    def reset_traces(self) -> None:
        for arr in (self.x1, self.x2, self.y1, self.y2):
            arr[:] = 0.0
