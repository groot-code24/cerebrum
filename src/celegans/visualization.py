"""Visualization utilities — matplotlib only."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np
from matplotlib.figure import Figure

from celegans.connectome import ConnectomeGraph
from celegans.neuron_types import NeuronType, classify_neuron
from celegans.utils.io import validate_path
from celegans.utils.logging import get_logger

logger = get_logger(__name__)

_TYPE_COLORS: Dict[str, str] = {
    NeuronType.SENSORY.value:     "#4fc3f7",
    NeuronType.INTERNEURON.value: "#aed581",
    NeuronType.MOTOR.value:       "#ef9a9a",
    NeuronType.PHARYNGEAL.value:  "#ce93d8",
    "unknown":                    "#b0bec5",
}


def _safe_save(fig: Figure, path: Path, project_root: Optional[Path] = None) -> None:
    root = project_root or path.parent
    safe_path = validate_path(path, root)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=safe_path.parent, suffix=".tmp.png")
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        os.replace(tmp, safe_path)
        logger.info("Figure saved: %s", safe_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def plot_connectome_graph(
    graph: ConnectomeGraph,
    activation: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    max_nodes_to_display: int = 150,
) -> Figure:
    """Render connectome as a force-directed graph."""
    import networkx as nx

    n = graph.data.num_nodes
    ei = graph.data.edge_index
    show_n = min(n, max_nodes_to_display)

    G = nx.DiGraph()
    G.add_nodes_from(range(show_n))
    for k in range(ei.shape[1]):
        s, t = int(ei[0, k]), int(ei[1, k])
        if s < show_n and t < show_n:
            G.add_edge(s, t)

    names_sub = graph.node_names[:show_n]
    pos = nx.spring_layout(G, seed=42, k=0.3)
    degrees = dict(G.degree())

    node_colors, node_sizes = [], []
    for i in G.nodes():
        name = names_sub[i] if i < len(names_sub) else "?"
        try:
            ntype = classify_neuron(name).value
        except KeyError:
            ntype = "unknown"
        color = _TYPE_COLORS.get(ntype, "#b0bec5")
        if activation is not None and i < len(activation):
            act_val = float(np.clip(activation[i], 0, 1))
            r_base = mcolors.to_rgb(color)
            warm = np.array([1.0, 0.4, 0.1])
            blended = tuple(float(r_base[c] * (1 - act_val) + warm[c] * act_val) for c in range(3))
            color = blended  # type: ignore[assignment]
        node_colors.append(color)
        node_sizes.append(30 + degrees.get(i, 0) * 5)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    edge_weights = []
    for u, v in G.edges():
        mask = (ei[0] == u) & (ei[1] == v)
        if mask.any() and graph.data.edge_attr is not None:
            edge_weights.append(float(graph.data.edge_attr[np.where(mask)[0][0], 0]))
        else:
            edge_weights.append(0.3)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=[min(w * 0.8 + 0.1, 0.8) for w in edge_weights],
        edge_color="#546e7a", width=0.4, arrows=True, arrowsize=5,
    )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.85)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=t)
        for t, c in _TYPE_COLORS.items() if t != "unknown"
    ]
    ax.legend(handles=legend_handles, loc="upper left", facecolor="#1a1a2e",
              labelcolor="white", fontsize=8)
    ax.set_title("C. elegans Connectome", color="white", fontsize=14, pad=10)
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        _safe_save(fig, output_path, project_root)
    return fig


def plot_spike_raster(
    spike_history: np.ndarray,
    node_names: List[str],
    sensory_indices: List[int],
    interneuron_indices: List[int],
    motor_indices: List[int],
    output_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    max_neurons: int = 100,
    dt: float = 0.1,
) -> Figure:
    """Standard neuroscience raster plot."""
    T = spike_history.shape[0]
    times = np.arange(T) * dt
    groups = [
        ("Sensory",     sensory_indices,     _TYPE_COLORS[NeuronType.SENSORY.value]),
        ("Interneuron", interneuron_indices, _TYPE_COLORS[NeuronType.INTERNEURON.value]),
        ("Motor",       motor_indices,       _TYPE_COLORS[NeuronType.MOTOR.value]),
    ]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    y_offset = 0
    y_ticks, y_labels = [], []

    for group_name, indices, color in groups:
        show = indices[:max_neurons]
        for j, ni in enumerate(show):
            if ni >= spike_history.shape[1]:
                continue
            spk_times = times[spike_history[:, ni].astype(bool)]
            ax.scatter(spk_times, np.full_like(spk_times, y_offset + j),
                       s=1.5, c=color, alpha=0.7, linewidths=0)
        if show:
            y_ticks.append(y_offset + len(show) // 2)
            y_labels.append(group_name)
        y_offset += max(len(show), 1) + 3

    ax.set_xlabel("Time (ms)", color="white", fontsize=11)
    ax.set_ylabel("Neuron group", color="white", fontsize=11)
    ax.set_title("Spike Raster", color="white", fontsize=13)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#546e7a")
    fig.tight_layout()
    if output_path:
        _safe_save(fig, output_path, project_root)
    return fig


def plot_trajectory(
    trajectory: np.ndarray,
    food_position: np.ndarray,
    output_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Figure:
    """2D worm head trajectory coloured blue→red over time."""
    T = trajectory.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    cmap = plt.get_cmap("coolwarm")

    for t in range(T - 1):
        color = cmap(t / max(T - 1, 1))
        ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1],
                "-", color=color, linewidth=1.2, alpha=0.8)

    ax.plot(*trajectory[0],  "o", color="#00e5ff", markersize=10, label="Start", zorder=5)
    ax.plot(*trajectory[-1], "s", color="#ff1744", markersize=10, label="End",   zorder=5)
    ax.plot(*food_position,  "*", color="#ffd600", markersize=16, label="Food",  zorder=6)

    ax.set_xlabel("x (a.u.)", color="white")
    ax.set_ylabel("y (a.u.)", color="white")
    ax.set_title("Worm Head Trajectory", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#546e7a")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Timestep", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout()
    if output_path:
        _safe_save(fig, output_path, project_root)
    return fig


def plot_ablation_results(
    results: list,
    output_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Figure:
    """Grouped bar chart: locomotion vs chemotaxis per ablation condition."""
    labels, loco, chem = [], [], []
    for r in results:
        label = (", ".join(r.ablated_neurons[:2]) if r.ablated_neurons
                 else f"Rnd {int(r.ablation_fraction * 100)}% (s{r.ablation_seed})")
        labels.append(label)
        loco.append(r.locomotion_score)
        chem.append(r.chemotaxis_score)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(x - w / 2, loco, w, label="Locomotion", color="#4fc3f7", alpha=0.85)
    ax.bar(x + w / 2, chem, w, label="Chemotaxis",  color="#ef9a9a", alpha=0.85)
    ax.axhline(1.0, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", color="white", fontsize=8)
    ax.set_ylabel("Normalised Score", color="white")
    ax.set_title("Ablation Results: Locomotion vs Chemotaxis", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#546e7a")
    fig.tight_layout()
    if output_path:
        _safe_save(fig, output_path, project_root)
    return fig


def render_episode_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 10,
    project_root: Optional[Path] = None,
) -> None:
    """Save RGB frames as MP4 using FFMpegWriter."""
    if not frames:
        raise ValueError("frames list is empty")
    root = project_root or output_path.parent
    safe_path = validate_path(output_path, root)
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFMpegWriter requires ffmpeg to be installed and on PATH. "
            "Install it with: sudo apt install ffmpeg  (Linux) or  brew install ffmpeg  (macOS)"
        )
    H, W = frames[0].shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i: int) -> list:
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames),
                                   interval=1000 // fps, blit=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(str(safe_path), writer=writer)
    plt.close(fig)
    logger.info("Video saved: %s (%d frames)", safe_path, len(frames))
