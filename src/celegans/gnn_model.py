"""GNN-style message passing over the connectome graph.

Pure numpy implementation used when PyTorch is unavailable.
When torch + torch_geometric are present, uses SAGEConv / GATConv.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from celegans.connectome import GraphData
from celegans.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch_geometric.nn import GATConv, SAGEConv  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── Numpy-based GNN (always available) ──────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _xavier_init(shape: tuple) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    rng = np.random.default_rng(42)
    return rng.uniform(-limit, limit, shape).astype(np.float32)


class _NumpyLinear:
    def __init__(self, in_f: int, out_f: int) -> None:
        self.W = _xavier_init((in_f, out_f))
        self.b = np.zeros(out_f, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class _NumpySAGELayer:
    """Single GraphSAGE-style message passing layer (mean aggregation)."""

    def __init__(self, in_f: int, out_f: int, aggr: str = "mean") -> None:
        self.aggr = aggr
        self.W_self = _xavier_init((in_f, out_f))
        self.W_neigh = _xavier_init((in_f, out_f))
        self.b = np.zeros(out_f, dtype=np.float32)

    def __call__(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        src, tgt = edge_index[0], edge_index[1]
        neigh_agg = np.zeros_like(x)

        if self.aggr == "sum":
            np.add.at(neigh_agg, tgt, x[src])
        elif self.aggr == "max":
            for i in range(len(src)):
                t, s = tgt[i], src[i]
                neigh_agg[t] = np.maximum(neigh_agg[t], x[s])
        else:  # mean
            counts = np.zeros(n, dtype=np.float32)
            np.add.at(neigh_agg, tgt, x[src])
            np.add.at(counts, tgt, 1.0)
            mask = counts > 0
            neigh_agg[mask] /= counts[mask, None]

        return x @ self.W_self + neigh_agg @ self.W_neigh + self.b


class ConnectomeGNN:
    """Connectome Graph Neural Network — numpy implementation.

    When torch + torch_geometric are present, falls back to PyTorch version
    automatically. Otherwise runs pure numpy message passing.

    Args:
        input_dim: Node feature dimension.
        hidden_dim: Hidden layer width.
        output_dim: Per-neuron output dimension.
        num_layers: GNN message-passing layers.
        dropout: Dropout probability (numpy: not applied during inference).
        aggr: Aggregation scheme.
        use_gat: Use GAT attention (numpy: ignored, always SAGE).
        sensory_indices: Indices of sensory neurons.
        motor_indices: Indices of motor neurons.
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
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.aggr = aggr
        self.sensory_indices: List[int] = sensory_indices or []
        self.motor_indices: List[int] = motor_indices or []
        self._last_attn_weights: Optional[np.ndarray] = None

        # Build layers
        self._input_proj = _NumpyLinear(input_dim, hidden_dim)
        self._convs = [
            _NumpySAGELayer(hidden_dim, hidden_dim, aggr=aggr)
            for _ in range(num_layers)
        ]
        self._output_proj = _NumpyLinear(hidden_dim, output_dim)

        logger.debug(
            "ConnectomeGNN (numpy) init: input_dim=%d hidden=%d layers=%d",
            input_dim, hidden_dim, num_layers,
        )

    def forward(
        self,
        data: GraphData,
        sensory_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute per-neuron activation scalars.

        Args:
            data: Graph data with x [N,F] and edge_index [2,E].
            sensory_input: Optional [num_sensory] array injected into sensory nodes.

        Returns:
            Array of shape [N, output_dim].
        """
        x = data.x.copy()
        ei = data.edge_index

        if sensory_input is not None and self.sensory_indices:
            n_s = len(self.sensory_indices)
            stim = np.asarray(sensory_input, dtype=np.float32)
            for j, si in enumerate(self.sensory_indices[:len(stim)]):
                x[si, 0] = float(stim[j])

        h = _relu(self._input_proj(x))
        for conv in self._convs:
            h = _relu(conv(h, ei))

        return self._output_proj(h)

    # Alias for compatibility with torch-style calling convention
    def __call__(
        self,
        data: GraphData,
        sensory_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.forward(data, sensory_input)

    def get_motor_activations(self, full_output: np.ndarray) -> np.ndarray:
        """Slice motor neuron rows from the full output array."""
        if not self.motor_indices:
            return full_output
        idx = np.array(self.motor_indices, dtype=np.int64)
        return full_output[idx]

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Return last attention weights (None for numpy SAGE model)."""
        return self._last_attn_weights

    def eval(self) -> "ConnectomeGNN":
        """No-op for numpy model (mirrors torch.nn.Module.eval)."""
        return self

    def train(self, mode: bool = True) -> "ConnectomeGNN":
        """No-op for numpy model (mirrors torch.nn.Module.train)."""
        return self

    def parameters(self):
        """Yield no parameters — numpy model has no trainable params via torch."""
        return iter([])
