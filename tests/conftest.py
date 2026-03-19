"""Shared helpers for test suite. Works with both pytest and unittest."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def make_mock_graph(n_nodes=10, n_edges=20, seed=0):
    from celegans.connectome import build_mock_connectome
    return build_mock_connectome(n_nodes=n_nodes, n_edges=n_edges, seed=seed)


def make_mock_gnn(graph):
    from celegans.gnn_model import ConnectomeGNN
    return ConnectomeGNN(
        input_dim=graph.data.x.shape[1],
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        sensory_indices=graph.get_sensory_indices(),
        motor_indices=graph.get_motor_indices(),
    )


def make_mock_lif(graph):
    from celegans.lif_neurons import LIFNeuronBank
    n = graph.data.num_nodes
    return LIFNeuronBank(
        n_total=n,
        sensory_indices=graph.get_sensory_indices(),
        interneuron_indices=graph.get_interneuron_indices(),
        motor_indices=graph.get_motor_indices(),
        tau_sensory=10.0, tau_interneuron=20.0, tau_motor=15.0,
        dt=0.1, threshold=1.0, reset_potential=0.0,
    )


def make_mock_env(graph):
    from celegans.environment import WormEnv
    n = graph.data.num_nodes
    motor_idx = graph.get_motor_indices()
    n_motor = max(len(motor_idx), 2)
    return WormEnv(
        n_neurons=n, body_segments=4, physics_substeps=2,
        food_gradient_strength=1.0, num_motor_neurons=n_motor, seed=42,
    )


def make_mock_runner(graph, tmp_path):
    from celegans.simulation import SimulationRunner
    gnn = make_mock_gnn(graph)
    lif = make_mock_lif(graph)
    env = make_mock_env(graph)
    return SimulationRunner(
        graph=graph, model=gnn, lif_bank=lif, env=env,
        sim_steps=10, results_dir=tmp_path, seed=42, project_root=tmp_path,
    )
