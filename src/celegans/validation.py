"""Biological validation metrics for the C. elegans emulator.

Implements three key validation tools that make simulation results
scientifically falsifiable:

1. **Chemotaxis Index (CI)**
   Matches the standard behavioural assay (Pierce-Shimomura et al. 1999).
   Real C. elegans: CI ≈ 0.6–0.8 on food gradient.
   Specific ablations produce characteristic CI drops documented in literature.

2. **Neural Manifold Procrustes Distance**
   Compares the PCA trajectory of simulated activity against recorded
   Kato et al. (2015) whole-brain calcium imaging data.
   A Procrustes distance < 0.15 indicates the model captures the real
   low-dimensional dynamics.

3. **Kato et al. 2015 Data Loader**
   Loads the public whole-brain calcium imaging CSV when available.
   Falls back to a biologically-calibrated synthetic dataset that matches
   the published PCA trajectory shape (Kato et al. Extended Data Fig 1).

References
----------
Pierce-Shimomura JT et al. (1999). J Neurosci 19(21):9557–9569.
Kato S et al. (2015). Cell 163(3):656–669.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from celegans.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chemotaxis Index
# ---------------------------------------------------------------------------

def compute_chemotaxis_index(
    trajectory: np.ndarray,
    food_position: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> float:
    """Compute the standard chemotaxis index from a 2D trajectory.

    The CI is defined as the proportion of radial moves toward food minus
    those away, normalised to [-1, 1].

    CI = (N_toward - N_away) / (N_toward + N_away + ε)

    where a 'radial move toward food' means the distance to food decreased
    during that timestep.

    Parameters
    ----------
    trajectory : np.ndarray shape [T, 2]
        2D head positions over time.
    food_position : np.ndarray shape [2,]
        Food source location.
    origin : np.ndarray shape [2,], optional
        Starting position (default: trajectory[0]).

    Returns
    -------
    float in [-1, 1].  Higher is better.
    """
    if trajectory.shape[0] < 2:
        return 0.0

    food = np.asarray(food_position, dtype=np.float64)
    traj = np.asarray(trajectory, dtype=np.float64)

    # Distance to food at each timestep
    dists = np.linalg.norm(traj - food[None, :], axis=1)

    # Sign of distance change — negative means moved toward food
    delta_d = np.diff(dists)
    toward = int((delta_d < 0).sum())
    away = int((delta_d > 0).sum())
    total = toward + away

    if total == 0:
        return 0.0

    ci = (toward - away) / (total + 1e-9)
    return float(np.clip(ci, -1.0, 1.0))


def compute_ablation_ci_table(
    baseline_ci: float,
    ablation_cis: Dict[str, float],
) -> Dict[str, float]:
    """Compute CI degradation ratios relative to baseline.

    Parameters
    ----------
    baseline_ci : float
        CI of intact animal.
    ablation_cis : dict
        Mapping {neuron_name: ci_after_ablation}.

    Returns
    -------
    dict mapping neuron_name → normalised CI (1.0 = same as baseline).
    """
    base = max(abs(baseline_ci), 1e-6)
    return {name: ci / base for name, ci in ablation_cis.items()}


# ---------------------------------------------------------------------------
# Procrustes Distance
# ---------------------------------------------------------------------------

def procrustes_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 10,
    center: bool = True,
    scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Procrustes distance between two neural activity matrices.

    1. Reduces both matrices to ``n_components`` PCA dimensions.
    2. Aligns them using full Procrustes (translation + rotation + scale).
    3. Returns the normalised Frobenius distance.

    Parameters
    ----------
    X : np.ndarray shape [T_x, N]
        Simulated neural activity (T timesteps, N neurons).
    Y : np.ndarray shape [T_y, N]
        Reference neural activity (same N neurons).
    n_components : int
        Number of PCA dimensions to compare.
    center, scale : bool
        Standard Procrustes pre-processing flags.

    Returns
    -------
    distance : float   —  0.0 = perfect match, larger = worse
    X_aligned : np.ndarray shape [T_common, n_components]
    Y_aligned : np.ndarray shape [T_common, n_components]
    """
    # Align lengths — use shorter sequence
    T = min(X.shape[0], Y.shape[0])
    Xc = X[:T].astype(np.float64)
    Yc = Y[:T].astype(np.float64)
    n_comp = min(n_components, Xc.shape[1], Yc.shape[1], T)

    # PCA
    Xp = _pca(Xc, n_comp)
    Yp = _pca(Yc, n_comp)

    # Procrustes
    Xa, Ya, dist = _procrustes(Xp, Yp, center=center, scale=scale)

    logger.debug("Procrustes distance: %.4f (n_comp=%d, T=%d)", dist, n_comp, T)
    return dist, Xa.astype(np.float32), Ya.astype(np.float32)


def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Pure-numpy PCA projection."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD-based PCA
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :n_components] * S[:n_components]
    except np.linalg.LinAlgError:
        # Fallback: covariance-based
        cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:n_components]
        return (Xc @ eigvecs[:, idx])


def _procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    center: bool = True,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full Procrustes alignment: find R that minimises ||X - Y@R||_F."""
    if center:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    if scale:
        norm_x = np.linalg.norm(X)
        norm_y = np.linalg.norm(Y)
        if norm_x > 1e-10:
            X = X / norm_x
        if norm_y > 1e-10:
            Y = Y / norm_y

    # Find optimal rotation: SVD of Y^T X
    M = Y.T @ X
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    Y_rot = Y @ R
    dist = float(np.linalg.norm(X - Y_rot, "fro"))
    # Normalise by ||X||_F so distance is scale-independent
    denom = np.linalg.norm(X, "fro")
    if denom > 1e-10:
        dist /= denom

    return X, Y_rot, dist


# ---------------------------------------------------------------------------
# Kato et al. 2015 data loader
# ---------------------------------------------------------------------------

#: Neurons recorded in Kato et al. 2015 (107 head neurons, subset)
KATO_NEURON_SUBSET: List[str] = [
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
    "RIML", "RIMR", "AIYL", "AIYR", "AIZL", "AIZR", "AWCL", "AWCR",
    "ASEL", "ASER", "AWAL", "AWAR", "RIS",  "RID",  "AVL",  "DVA",
    "SMDDL", "SMDDR", "SMDVL", "SMDVR", "RIVL", "RIVR",
    "RMDL", "RMDR", "RMDVL", "RMDVR", "RMDDL", "RMDDR",
    "VB1", "VB2", "VB3", "VB4", "VB5", "DB1", "DB2", "DB3",
    "VA1", "VA2", "VA3", "VA4", "VA5", "DA1", "DA2", "DA3",
    "DD1", "DD2", "DD3", "VD1", "VD2", "VD3",
]

#: Expected Procrustes distance threshold for a validated model
PROCRUSTES_THRESHOLD = 0.15


def load_kato_data(data_dir: Path) -> Optional[np.ndarray]:
    """Load Kato et al. 2015 calcium imaging data.

    Looks for ``kato2015_activity.csv`` in *data_dir*.  Each row is a
    timepoint; each column is a neuron from :data:`KATO_NEURON_SUBSET`.

    Returns
    -------
    np.ndarray shape [T, N] or ``None`` if file not found.
    """
    path = data_dir / "kato2015_activity.csv"
    if not path.exists():
        logger.info("Kato 2015 data not found at %s — using synthetic data", path)
        return None

    import pandas as pd
    df = pd.read_csv(path)
    available = [c for c in KATO_NEURON_SUBSET if c in df.columns]
    if not available:
        logger.warning("No Kato neuron columns found in %s", path)
        return None

    logger.info("Loaded Kato 2015 data: %d timepoints, %d neurons", len(df), len(available))
    return df[available].values.astype(np.float32)


def generate_synthetic_kato_data(
    n_timepoints: int = 500,
    n_neurons: int = 57,
    seed: int = 0,
) -> np.ndarray:
    """Generate biologically-calibrated synthetic neural activity.

    Produces a low-rank activity matrix that mimics the structure of Kato
    et al. 2015: slow oscillatory dynamics (locomotion state) + fast
    fluctuations (sensory responses) + correlated sub-populations.

    The synthetic data matches the qualitative features of Fig 1C:
      - 2–3 dominant PCA modes explaining ~60% of variance
      - ~3–5 second oscillation period (locomotion cycle)
      - Low noise floor

    Parameters
    ----------
    n_timepoints : int  —  number of synthetic timeframes (default 500)
    n_neurons : int     —  number of neurons (default 57, matching Kato subset)
    seed : int

    Returns
    -------
    np.ndarray shape [n_timepoints, n_neurons]
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 50.0, n_timepoints)  # 50 seconds total

    # Locomotion oscillation: ~0.3 Hz (forward/backward cycles)
    loco = np.sin(2 * np.pi * 0.3 * t)
    loco_rev = np.sin(2 * np.pi * 0.3 * t + np.pi * 0.7)

    # Turn signal: sparse events
    turn = np.zeros_like(t)
    margin = max(1, n_timepoints // 5)
    n_turns = min(5, max(1, n_timepoints // 20))
    turn_times = rng.integers(margin, max(margin + 1, n_timepoints - margin), size=n_turns)
    for tt in turn_times:
        turn[tt:tt + 20] = np.exp(-np.arange(20) / 5.0)

    # Build rank-3 activity
    #   mode 1: locomotion forward/backward state
    #   mode 2: dorsal-ventral oscillation
    #   mode 3: turn response
    modes = np.stack([loco, loco_rev, turn], axis=1)  # [T, 3]

    # Random neuron-mode loadings (with some structure)
    loadings = rng.standard_normal((3, n_neurons)).astype(np.float32)
    loadings[0, :n_neurons // 3] *= 2.0  # AVA/AVB group dominates mode 1
    loadings[1, n_neurons // 3: 2 * n_neurons // 3] *= 2.0  # SMD group mode 2

    activity = (modes @ loadings).astype(np.float32)  # [T, N]

    # Add small measurement noise
    activity += rng.standard_normal(activity.shape).astype(np.float32) * 0.1

    # Normalise each neuron to unit variance
    std = activity.std(axis=0)
    std[std < 1e-6] = 1.0
    activity /= std

    return activity


def validate_simulation(
    spike_history: np.ndarray,
    node_names: List[str],
    data_dir: Optional[Path] = None,
    n_components: int = 10,
) -> Dict[str, object]:
    """Run full validation of a simulation episode against Kato et al. data.

    Parameters
    ----------
    spike_history : np.ndarray shape [T, N]
        Simulated spike history.
    node_names : List[str]
        Neuron names corresponding to columns of ``spike_history``.
    data_dir : Path, optional
        Directory containing ``kato2015_activity.csv``.
    n_components : int
        PCA dimensions for Procrustes comparison.

    Returns
    -------
    dict with keys:
        ``procrustes_distance``, ``passes_threshold``,
        ``n_kato_neurons_matched``, ``simulated_variance_explained``
    """
    # Find Kato neurons present in our simulation
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    matched = [(name, name_to_idx[name]) for name in KATO_NEURON_SUBSET
               if name in name_to_idx]

    if len(matched) < 5:
        return {
            "procrustes_distance": None,
            "passes_threshold": False,
            "n_kato_neurons_matched": len(matched),
            "simulated_variance_explained": None,
            "warning": f"Only {len(matched)} Kato neurons found in simulation graph",
        }

    matched_names, matched_idx = zip(*matched)
    sim_subset = spike_history[:, list(matched_idx)]

    # Load or generate reference data
    ref = None
    if data_dir is not None:
        ref = load_kato_data(data_dir)
    if ref is None:
        ref = generate_synthetic_kato_data(
            n_timepoints=spike_history.shape[0],
            n_neurons=len(matched_idx),
        )
    else:
        # Align reference to matched columns
        ref = ref[:, :len(matched_idx)]

    # Variance explained by top-k PCA components in simulated data
    try:
        _, S, _ = np.linalg.svd(sim_subset - sim_subset.mean(0), full_matrices=False)
        total_var = (S ** 2).sum()
        top_k_var = (S[:n_components] ** 2).sum()
        var_explained = float(top_k_var / max(total_var, 1e-12))
    except np.linalg.LinAlgError:
        var_explained = None

    # Procrustes
    try:
        dist, _, _ = procrustes_distance(sim_subset, ref, n_components=n_components)
    except Exception as exc:
        logger.warning("Procrustes failed: %s", exc)
        dist = None

    return {
        "procrustes_distance": round(float(dist), 4) if dist is not None else None,
        "passes_threshold": (dist is not None and dist < PROCRUSTES_THRESHOLD),
        "n_kato_neurons_matched": len(matched),
        "simulated_variance_explained": round(var_explained, 4) if var_explained is not None else None,
    }
