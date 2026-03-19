"""Temporal Graph Neural Network with axonal propagation delays.

Standard GNNs apply message passing instantaneously.  In the real nervous
system — including C. elegans — signals travel along axons at finite speed,
introducing per-synapse delays of ~1–20 ms.

This module implements a ring-buffer-based delayed GNN where each message
from neuron *i* to neuron *j* uses the spike state from *delay[i,j]* timesteps
ago.  When no per-edge delay is provided, delays are estimated from a simple
axon-length proxy derived from the graph topology.

Architecture
------------
TemporalConnectomeGNN wraps the standard ConnectomeGNN and adds:
  - A ring buffer of past spike/activation states
  - Edge-wise delay lookup (integer timesteps)
  - Delayed message aggregation

References
----------
Izhikevich EM (2006). Polychronization: computation with spikes. Neural Comput 18:245–282.
Bhattacharya S et al. (2019). PLoS Comput Biol 15:e1007279.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from celegans.connectome import GraphData
from celegans.gnn_model import ConnectomeGNN, _relu, _NumpySAGELayer, _NumpyLinear
from celegans.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_DELAY_STEPS = 50   # hard cap on delay ring buffer depth


def estimate_delays(
    edge_index: np.ndarray,
    n_nodes: int,
    dt: float = 0.1,
    conduction_velocity: float = 0.3,  # mm/ms — C. elegans axon speed
    body_length_mm: float = 1.0,
) -> np.ndarray:
    """Estimate per-edge propagation delay in integer timesteps.

    Uses shortest-path distances as a proxy for axon length, normalised to
    the estimated body length. Real C. elegans axonal delay data from
    Bhattacharya et al. ranges from ~2–18 ms.

    Parameters
    ----------
    edge_index : np.ndarray shape [2, E]
    n_nodes : int
    dt : float  —  simulation timestep in ms
    conduction_velocity : float  —  mm/ms (default 0.3 for C. elegans)
    body_length_mm : float  —  normalisation factor

    Returns
    -------
    delays : np.ndarray[int32, E]  —  delay in timesteps (≥ 1)
    """
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return np.ones(0, dtype=np.int32)

    # Use graph hop count as distance proxy — scale to mm
    # Max anatomical distance ≈ body_length_mm
    # We normalise max hop count → body_length_mm
    from collections import defaultdict

    # Build adjacency list
    adj: dict = defaultdict(list)
    for e in range(n_edges):
        s, t = int(edge_index[0, e]), int(edge_index[1, e])
        adj[s].append(t)

    # BFS from each node (capped at n_nodes for speed)
    hop_matrix = np.full((n_nodes, n_nodes), n_nodes, dtype=np.int32)
    np.fill_diagonal(hop_matrix, 0)

    for start in range(n_nodes):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if nbr not in visited:
                    visited[nbr] = visited[node] + 1
                    hop_matrix[start, nbr] = visited[nbr]
                    queue.append(nbr)

    max_hop = max(hop_matrix.max(), 1)

    src = edge_index[0]
    tgt = edge_index[1]
    hops = hop_matrix[src, tgt]

    # Scale: max_hop → body_length_mm
    length_mm = (hops / max_hop) * body_length_mm
    delay_ms = length_mm / conduction_velocity
    delay_steps = np.maximum(1, np.round(delay_ms / dt).astype(np.int32))
    delay_steps = np.minimum(delay_steps, _MAX_DELAY_STEPS)

    logger.debug(
        "Delay estimation: min=%d max=%d mean=%.1f timesteps",
        delay_steps.min(), delay_steps.max(), delay_steps.mean(),
    )
    return delay_steps


class TemporalConnectomeGNN(ConnectomeGNN):
    """ConnectomeGNN extended with per-edge axonal propagation delays.

    Maintains a ring buffer of past activation states.  During forward pass,
    messages from neuron *src* to neuron *tgt* use the activation state
    ``delay[src, tgt]`` timesteps ago.

    Parameters
    ----------
    edge_delays : np.ndarray[int32, E], optional
        Per-edge delay in timesteps.  If ``None``, estimated automatically
        from graph topology via :func:`estimate_delays`.
    max_delay : int
        Ring buffer depth.  Must be ≥ ``edge_delays.max()``.
    All other parameters are forwarded to :class:`ConnectomeGNN`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggr: str = "mean",
        use_gat: bool = False,
        sensory_indices: Optional[List[int]] = None,
        motor_indices: Optional[List[int]] = None,
        edge_delays: Optional[np.ndarray] = None,
        max_delay: int = _MAX_DELAY_STEPS,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggr=aggr,
            use_gat=use_gat,
            sensory_indices=sensory_indices,
            motor_indices=motor_indices,
        )
        self._edge_delays: Optional[np.ndarray] = edge_delays
        self._max_delay = max_delay
        # Ring buffer: deque of [N, hidden_dim] arrays
        self._activation_buffer: deque = deque(maxlen=max_delay + 1)

    def _ensure_delays(self, data: GraphData) -> np.ndarray:
        """Lazily estimate delays if not provided."""
        if self._edge_delays is None:
            self._edge_delays = estimate_delays(
                data.edge_index, data.num_nodes
            )
        return self._edge_delays

    def forward(
        self,
        data: GraphData,
        sensory_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Temporal forward pass with delayed message passing.

        Parameters
        ----------
        data : GraphData
        sensory_input : optional [num_sensory] array

        Returns
        -------
        np.ndarray shape [N, output_dim]
        """
        delays = self._ensure_delays(data)
        x = data.x.copy()
        ei = data.edge_index

        if sensory_input is not None and self.sensory_indices:
            stim = np.asarray(sensory_input, dtype=np.float32)
            for j, si in enumerate(self.sensory_indices[:len(stim)]):
                x[si, 0] = float(stim[j])

        h = _relu(self._input_proj(x))

        # Push current h into buffer
        self._activation_buffer.append(h.copy())

        # Temporal SAGE layers
        for conv in self._convs:
            h = self._temporal_sage(h, ei, delays, conv)
            h = _relu(h)

        return self._output_proj(h)

    def _temporal_sage(
        self,
        h: np.ndarray,
        edge_index: np.ndarray,
        delays: np.ndarray,
        conv,
    ) -> np.ndarray:
        """SAGE aggregation using time-delayed source activations."""
        n, d = h.shape
        neigh_agg = np.zeros((n, d), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        src = edge_index[0]
        tgt = edge_index[1]

        buf_len = len(self._activation_buffer)

        for e_idx in range(len(src)):
            s, t = int(src[e_idx]), int(tgt[e_idx])
            delay = int(delays[e_idx])
            # Look back 'delay' steps in buffer
            buf_idx = buf_len - 1 - delay
            if buf_idx >= 0:
                past_h = self._activation_buffer[buf_idx]
            else:
                past_h = h  # not enough history yet — use current
            np.add.at(neigh_agg, t, past_h[s])
            counts[t] += 1.0

        # Mean aggregation
        mask = counts > 0
        neigh_agg[mask] /= counts[mask, None]

        return h @ conv.W_self + neigh_agg @ conv.W_neigh + conv.b

    def reset_buffer(self) -> None:
        """Clear the activation history buffer (call at episode start)."""
        self._activation_buffer.clear()
