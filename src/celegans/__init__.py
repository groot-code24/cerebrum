"""C. elegans Connectome Emulator package."""

__version__ = "0.2.0"
__author__ = "C. elegans Emulator Contributors"

# Core modules (always available)
from celegans import (
    config,
    connectome,
    neuron_types,
    lif_neurons,
    adex_neurons,
    gnn_model,
    temporal_gnn,
    environment,
    simulation,
    ablation,
    visualization,
    validation,
    tracking,
    graph_vae,
    stdp,
)

__all__ = [
    "config", "connectome", "neuron_types",
    "lif_neurons", "adex_neurons",
    "gnn_model", "temporal_gnn",
    "environment", "simulation", "ablation",
    "visualization", "validation", "tracking",
    "graph_vae", "stdp",
]
