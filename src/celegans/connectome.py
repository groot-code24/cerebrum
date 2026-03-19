"""Load and parse the C. elegans connectome into a graph data structure.

Uses numpy arrays throughout. PyTorch Geometric is optional — when absent,
a lightweight numpy-based Graph object is used instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from celegans.neuron_types import NeuronType, classify_neuron
from celegans.utils.logging import get_logger

logger = get_logger(__name__)

_COL_FROM = "Origin"
_COL_TO = "Target"
_COL_WEIGHT = "Number"
_EXPECTED_COLUMNS = {_COL_FROM, _COL_TO, _COL_WEIGHT}

# Column alias sets matching all known OpenWorm CSV formats
_ORIGIN_ALIASES = {
    "origin", "pre", "from", "neuron1", "neuron 1", "source",
    "preneuroname", "sending cell body",
}
_TARGET_ALIASES = {
    "target", "post", "to", "neuron2", "neuron 2", "destination",
    "postneuroname", "receiving cell body",
}
_NUMBER_ALIASES = {
    "number", "sections", "count", "weight", "synapses", "n",
    "number of gap junctions", "nbconnections", "strength",
}


def _normalize_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename columns from any OpenWorm format to canonical Origin/Target/Number.

    Handles c302 format (Pre/Post/Sections), gap junction format
    (Neuron 1 / Neuron 2 / Number of Gap Junctions), WormAtlas format
    (Origin/Target/Number), and variants thereof.
    """
    cols = list(df.columns)
    if _EXPECTED_COLUMNS.issubset(set(cols)):
        return df  # Already canonical

    rename: Dict[str, str] = {}
    matched: Dict[str, Optional[str]] = {"Origin": None, "Target": None, "Number": None}

    for col in cols:
        key = col.strip().lower()
        if matched["Origin"] is None and key in _ORIGIN_ALIASES:
            matched["Origin"] = col
            rename[col] = "Origin"
        elif matched["Target"] is None and key in _TARGET_ALIASES:
            matched["Target"] = col
            rename[col] = "Target"
        elif matched["Number"] is None and key in _NUMBER_ALIASES:
            matched["Number"] = col
            rename[col] = "Number"

    unresolved = [k for k, v in matched.items() if v is None]
    if unresolved:
        raise ValueError(
            f"[{source}] Cannot map columns {cols} to canonical names. "
            f"Unresolved: {unresolved}. "
            f"Expected one of: Origin/Pre/From, Target/Post/To, "
            f"Number/Sections/Count/Weight."
        )

    if rename:
        logger.info("[%s] Normalizing columns: %s", source, rename)
        df = df.rename(columns=rename)

    return df


def _validate_dataframe(df: pd.DataFrame, source: str) -> None:
    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{source}] Missing columns after normalization: {missing}")
    if df[_COL_FROM].isna().any() or df[_COL_TO].isna().any():
        raise ValueError(f"[{source}] NaN found in neuron name columns")
    if not pd.api.types.is_numeric_dtype(df[_COL_WEIGHT]):
        raise ValueError(f"[{source}] Column '{_COL_WEIGHT}' must be numeric")
    if df[_COL_WEIGHT].isna().any():
        raise ValueError(f"[{source}] NaN in weight column")
    if (df[_COL_WEIGHT] < 0).any():
        raise ValueError(f"[{source}] Negative synapse counts found")


# ─── Lightweight graph data container ────────────────────────────────────────

@dataclass
class GraphData:
    """Numpy-based graph data (mirrors torch_geometric.data.Data interface)."""
    x: np.ndarray           # [N, F] node feature matrix
    edge_index: np.ndarray  # [2, E] source/target indices (int)
    edge_attr: np.ndarray   # [E, 1] edge weights

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# Try to provide torch_geometric Data as an alias when available
try:
    from torch_geometric.data import Data as _PyGData  # type: ignore
    import torch as _torch

    def _to_pyg(gd: GraphData) -> _PyGData:
        return _PyGData(
            x=_torch.tensor(gd.x, dtype=_torch.float),
            edge_index=_torch.tensor(gd.edge_index, dtype=_torch.long),
            edge_attr=_torch.tensor(gd.edge_attr, dtype=_torch.float),
        )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    def _to_pyg(gd: GraphData) -> GraphData:  # type: ignore[misc]
        return gd


# ─── ConnectomeGraph ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConnectomeGraph:
    """Immutable representation of the C. elegans connectome."""

    data: GraphData
    node_names: List[str]
    _sensory_idx: List[int] = field(default_factory=list, compare=False)
    _motor_idx: List[int] = field(default_factory=list, compare=False)
    _interneuron_idx: List[int] = field(default_factory=list, compare=False)

    def get_sensory_indices(self) -> List[int]:
        return list(self._sensory_idx)

    def get_motor_indices(self) -> List[int]:
        return list(self._motor_idx)

    def get_interneuron_indices(self) -> List[int]:
        return list(self._interneuron_idx)

    def ablate_neurons(self, neuron_names: List[str]) -> "ConnectomeGraph":
        """Return new graph with named neurons and their edges removed."""
        ablate_set = set(neuron_names)
        keep_mask = np.array([n not in ablate_set for n in self.node_names])
        keep_indices = np.where(keep_mask)[0]

        index_map = np.full(len(self.node_names), -1, dtype=np.int64)
        index_map[keep_indices] = np.arange(len(keep_indices), dtype=np.int64)

        ei = self.data.edge_index  # [2, E]
        src_keep = keep_mask[ei[0]]
        tgt_keep = keep_mask[ei[1]]
        edge_mask = src_keep & tgt_keep

        new_ei = index_map[ei[:, edge_mask]]
        new_ea = self.data.edge_attr[edge_mask]
        new_x = self.data.x[keep_mask]
        new_names = [n for n in self.node_names if n not in ablate_set]

        new_data = GraphData(x=new_x, edge_index=new_ei, edge_attr=new_ea)
        return _build_graph_from_components(new_data, new_names)

    def ablate_random_synapses(self, fraction: float, seed: int) -> "ConnectomeGraph":
        """Return new graph with a random fraction of edges removed."""
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        rng = np.random.default_rng(seed)
        n_edges = self.data.num_edges
        n_remove = int(n_edges * fraction)
        remove_idx = rng.choice(n_edges, size=n_remove, replace=False)
        keep_mask = np.ones(n_edges, dtype=bool)
        keep_mask[remove_idx] = False
        new_ei = self.data.edge_index[:, keep_mask]
        new_ea = self.data.edge_attr[keep_mask]
        new_data = GraphData(x=self.data.x, edge_index=new_ei, edge_attr=new_ea)
        return _build_graph_from_components(new_data, list(self.node_names))

    def summary(self) -> Dict[str, object]:
        n = self.data.num_nodes
        e = self.data.num_edges
        density = e / (n * (n - 1)) if n > 1 else 0.0
        avg_deg = 2 * e / n if n > 0 else 0.0
        return {
            "num_nodes": n,
            "num_edges": e,
            "density": round(density, 6),
            "avg_degree": round(avg_deg, 3),
            "num_sensory": len(self._sensory_idx),
            "num_motor": len(self._motor_idx),
            "num_interneuron": len(self._interneuron_idx),
        }


# ─── Factory helpers ──────────────────────────────────────────────────────────

_TYPE_ORDER = [
    NeuronType.SENSORY, NeuronType.INTERNEURON, NeuronType.MOTOR, NeuronType.PHARYNGEAL
]
_TYPE_INDEX: Dict[NeuronType, int] = {t: i for i, t in enumerate(_TYPE_ORDER)}


def _build_graph_from_components(
    data: GraphData, node_names: List[str]
) -> ConnectomeGraph:
    sensory_idx, motor_idx, interneuron_idx = [], [], []
    for i, name in enumerate(node_names):
        try:
            ntype = classify_neuron(name)
        except KeyError:
            ntype = NeuronType.INTERNEURON
        if ntype == NeuronType.SENSORY:
            sensory_idx.append(i)
        elif ntype == NeuronType.MOTOR:
            motor_idx.append(i)
        elif ntype == NeuronType.INTERNEURON:
            interneuron_idx.append(i)
    return ConnectomeGraph(
        data=data,
        node_names=node_names,
        _sensory_idx=sensory_idx,
        _motor_idx=motor_idx,
        _interneuron_idx=interneuron_idx,
    )


def load_connectome(data_dir: Path) -> ConnectomeGraph:
    """Load C. elegans connectome CSVs from data_dir."""
    chem_path = data_dir / "connectome_chemical.csv"
    gap_path = data_dir / "connectome_gap.csv"

    if not chem_path.exists():
        raise FileNotFoundError(
            f"Chemical synapse CSV not found at {chem_path}. "
            "Run `python data/download.py` first."
        )

    logger.info("Loading chemical synapse data from %s", chem_path)
    chem_df = pd.read_csv(chem_path)
    chem_df = _normalize_dataframe(chem_df, str(chem_path))
    _validate_dataframe(chem_df, str(chem_path))

    frames = [chem_df]
    if gap_path.exists():
        logger.info("Loading gap junction data from %s", gap_path)
        gap_df = pd.read_csv(gap_path)
        gap_df = _normalize_dataframe(gap_df, str(gap_path))
        _validate_dataframe(gap_df, str(gap_path))
        frames.append(gap_df)

    df = pd.concat(frames, ignore_index=True)

    all_names: List[str] = list(
        dict.fromkeys(df[_COL_FROM].tolist() + df[_COL_TO].tolist())
    )
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(all_names)}
    n_nodes = len(all_names)

    in_degree = np.zeros(n_nodes, dtype=np.float32)
    out_degree = np.zeros(n_nodes, dtype=np.float32)
    src_idx = np.array([name_to_idx[n] for n in df[_COL_FROM]], dtype=np.int64)
    tgt_idx = np.array([name_to_idx[n] for n in df[_COL_TO]], dtype=np.int64)
    np.add.at(out_degree, src_idx, 1.0)
    np.add.at(in_degree, tgt_idx, 1.0)

    max_deg = max(in_degree.max(), out_degree.max(), 1.0)
    in_degree /= max_deg
    out_degree /= max_deg

    type_onehot = np.zeros((n_nodes, len(_TYPE_ORDER)), dtype=np.float32)
    for i, name in enumerate(all_names):
        try:
            ntype = classify_neuron(name)
        except KeyError:
            ntype = NeuronType.INTERNEURON
        type_onehot[i, _TYPE_INDEX[ntype]] = 1.0

    x = np.concatenate(
        [type_onehot, in_degree[:, None], out_degree[:, None]], axis=1
    ).astype(np.float32)

    edge_index = np.stack([src_idx, tgt_idx], axis=0)

    weights = df[_COL_WEIGHT].values.astype(np.float32)
    w_max = weights.max()
    if w_max > 0:
        weights /= w_max
    edge_attr = weights[:, None]

    data = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph = _build_graph_from_components(data, all_names)
    logger.info("Connectome loaded: %s", graph.summary())
    return graph


def build_mock_connectome(
    n_nodes: int = 10,
    n_edges: int = 20,
    seed: int = 0,
) -> ConnectomeGraph:
    """Build a small synthetic connectome for unit testing."""
    from celegans.neuron_types import NEURON_REGISTRY
    rng = np.random.default_rng(seed)
    all_real = list(NEURON_REGISTRY.keys())[:n_nodes]
    n = len(all_real)

    edges_set: set = set()
    attempts = 0
    while len(edges_set) < n_edges and attempts < n_edges * 20:
        s = int(rng.integers(0, n))
        t = int(rng.integers(0, n))
        if s != t:
            edges_set.add((s, t))
        attempts += 1

    edges = list(edges_set)
    if not edges:
        edges = [(0, 1)]

    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
    edge_attr = rng.random((len(edges), 1)).astype(np.float32)
    x = rng.random((n, 6)).astype(np.float32)

    data = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return _build_graph_from_components(data, all_real)
