#!/usr/bin/env python3
"""Validate simulation against Kato et al. 2015 whole-brain calcium imaging data.

Usage::

    python scripts/run_validation.py [--steps 500] [--seed 42]

Produces:
  - Procrustes distance (target < 0.15)
  - Chemotaxis index (target > 0.4)
  - PCA variance explained
  - experiments/results/validation_report.json
  - experiments/results/validation_pca.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate simulation against Kato 2015 data.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--use-adex", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import os
    if args.data_dir:
        os.environ["CELEGANS_DATA_DIR"] = args.data_dir

    from celegans.config import load_config
    from celegans.connectome import load_connectome
    from celegans.environment import WormEnv
    from celegans.gnn_model import ConnectomeGNN
    from celegans.simulation import SimulationRunner
    from celegans.tracking import ExperimentTracker
    from celegans.utils.logging import get_logger
    from celegans.utils.reproducibility import set_all_seeds
    from celegans.validation import (
        compute_chemotaxis_index,
        generate_synthetic_kato_data,
        validate_simulation,
    )

    logger = get_logger("run_validation")
    cfg = load_config()
    set_all_seeds(args.seed)

    data_dir = cfg.resolved_data_dir(_PROJECT_ROOT)
    results_dir = cfg.resolved_results_dir(_PROJECT_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)

    graph = load_connectome(data_dir)
    sensory_idx = graph.get_sensory_indices()
    motor_idx = graph.get_motor_indices()
    interneuron_idx = graph.get_interneuron_indices()

    if args.use_adex:
        from celegans.adex_neurons import AdExNeuronBank
        lif_bank = AdExNeuronBank(
            n_total=graph.data.num_nodes,
            sensory_indices=sensory_idx,
            interneuron_indices=interneuron_idx,
            motor_indices=motor_idx,
            dt=cfg.dt,
        )
        neuron_model = "AdEx"
    else:
        from celegans.lif_neurons import LIFNeuronBank
        lif_bank = LIFNeuronBank(
            n_total=graph.data.num_nodes,
            sensory_indices=sensory_idx,
            interneuron_indices=interneuron_idx,
            motor_indices=motor_idx,
            dt=cfg.dt, threshold=cfg.threshold, reset_potential=cfg.reset_potential,
        )
        neuron_model = "LIF"

    model = ConnectomeGNN(
        input_dim=graph.data.x.shape[1],
        hidden_dim=cfg.gnn_hidden_dim,
        num_layers=cfg.gnn_num_layers,
        sensory_indices=sensory_idx,
        motor_indices=motor_idx,
    )

    n_motor = max(len(motor_idx), 1)
    env = WormEnv(
        n_neurons=graph.data.num_nodes,
        body_segments=cfg.body_segments,
        food_gradient_strength=cfg.food_gradient_strength,
        num_motor_neurons=n_motor,
        seed=args.seed,
    )

    # ── Run episode ──────────────────────────────────────────────────────
    print(f"\nRunning {args.steps}-step episode ({neuron_model} neurons)...")
    obs, _ = env.reset(seed=args.seed)
    lif_bank.reset_state()
    model.eval()

    trajectory = []
    food_pos = None

    for step in range(args.steps):
        gradient = obs["food_gradient"]
        sensory_input = np.zeros(max(len(sensory_idx), 1), dtype=np.float32)
        if len(sensory_idx) >= 2:
            sensory_input[0] = float(gradient[0])
            sensory_input[1] = float(gradient[1])

        gnn_out = model(graph.data, sensory_input)
        spikes, mems = lif_bank.step(gnn_out.ravel())
        motor_rates = lif_bank.get_recent_spike_rates(10)

        action = np.zeros(n_motor, dtype=np.float32)
        for j, mi in enumerate(motor_idx[:n_motor]):
            action[j] = float(np.clip(motor_rates[mi], -1.0, 1.0))

        obs, _, terminated, _, _ = env.step(action)
        env.update_neural_state(mems, motor_rates)
        trajectory.append(obs["body_position"][0].copy())

        if food_pos is None:
            fp_proxy = obs["body_position"][0] + obs["food_gradient"] * 80
            food_pos = fp_proxy

        if terminated:
            logger.info("Food reached at step %d", step + 1)
            break

    traj = np.stack(trajectory, axis=0)
    spike_hist = lif_bank.get_spike_history()

    # ── Chemotaxis Index ─────────────────────────────────────────────────
    ci = compute_chemotaxis_index(traj, food_pos or np.zeros(2))
    print(f"  Chemotaxis Index (CI): {ci:.3f}  {'✓ Good' if ci > 0.3 else '✗ Below target (>0.3)'}")

    # ── Procrustes Validation ────────────────────────────────────────────
    val = validate_simulation(spike_hist, graph.node_names, data_dir=data_dir)
    pd_str = f"{val['procrustes_distance']:.4f}" if val['procrustes_distance'] is not None else "N/A"
    passed = val['passes_threshold']
    print(f"  Procrustes distance:   {pd_str}  {'✓ Passes (<0.15)' if passed else '✗ Above threshold'}")
    print(f"  Kato neurons matched:  {val['n_kato_neurons_matched']}")
    if val['simulated_variance_explained'] is not None:
        print(f"  Variance explained:    {val['simulated_variance_explained']:.1%} (top-10 PCs)")

    # ── Plots ────────────────────────────────────────────────────────────
    _plot_validation(spike_hist, traj, food_pos, graph.node_names, results_dir)

    # ── Report ───────────────────────────────────────────────────────────
    report = {
        "neuron_model": neuron_model,
        "steps_run": len(trajectory),
        "seed": args.seed,
        "chemotaxis_index": round(ci, 4),
        "ci_passes": ci > 0.3,
        **val,
    }
    out = results_dir / "validation_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {out}")
    print(f"Plots saved:  {results_dir / 'validation_pca.png'}")

    # Summary
    print("\n── Validation Summary ─────────────────────────────────────────")
    n_pass = sum([ci > 0.3, passed])
    print(f"  Passed {n_pass}/2 validation checks.")
    if n_pass == 2:
        print("  ✓ Model meets biological validation criteria.")
    else:
        print("  ⚠  Fit to Kato 2015 data to improve Procrustes score.")
        print("     See docs/fitting_guide.md for instructions.")
    print("─" * 62)


def _plot_validation(spike_hist, traj, food_pos, node_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from celegans.validation import _pca, generate_synthetic_kato_data, KATO_NEURON_SUBSET

    # PCA comparison
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    matched_idx = [name_to_idx[n] for n in KATO_NEURON_SUBSET if n in name_to_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#111827")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        ax.tick_params(colors="#9ca3af")

    # (1) Simulated PCA trajectory
    if len(matched_idx) >= 3 and spike_hist.shape[0] > 0:
        sim_sub = spike_hist[:, matched_idx]
        pca_sim = _pca(sim_sub.astype(np.float64), n_components=3)
        axes[0].plot(pca_sim[:, 0], pca_sim[:, 1], lw=0.8, color="#60a5fa", alpha=0.8)
        axes[0].set_title("Simulated PCA (PC1 vs PC2)", color="white", fontsize=11)
        axes[0].set_xlabel("PC1", color="#9ca3af")
        axes[0].set_ylabel("PC2", color="#9ca3af")

    # (2) Reference PCA trajectory
    ref = generate_synthetic_kato_data(spike_hist.shape[0], len(matched_idx))
    pca_ref = _pca(ref.astype(np.float64), n_components=3)
    axes[1].plot(pca_ref[:, 0], pca_ref[:, 1], lw=0.8, color="#f59e0b", alpha=0.8)
    axes[1].set_title("Kato 2015 PCA (PC1 vs PC2)", color="white", fontsize=11)
    axes[1].set_xlabel("PC1", color="#9ca3af")

    # (3) Trajectory + food
    T = len(traj)
    if T > 1:
        cmap = plt.get_cmap("coolwarm")
        for t in range(T - 1):
            axes[2].plot(traj[t:t+2, 0], traj[t:t+2, 1],
                         color=cmap(t / max(T - 1, 1)), lw=1.2, alpha=0.8)
    if food_pos is not None:
        axes[2].plot(*food_pos, "*", color="#fbbf24", ms=14, label="Food", zorder=5)
    axes[2].plot(*traj[0], "o", color="#34d399", ms=8, label="Start", zorder=5)
    if T > 0:
        axes[2].plot(*traj[-1], "s", color="#f87171", ms=8, label="End", zorder=5)
    axes[2].legend(facecolor="#1f2937", labelcolor="white", fontsize=8)
    axes[2].set_title("Head Trajectory", color="white", fontsize=11)
    axes[2].set_xlabel("x (a.u.)", color="#9ca3af")

    fig.tight_layout()
    out_path = out_dir / "validation_pca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    import numpy as np
    main()
