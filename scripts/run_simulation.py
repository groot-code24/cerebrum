#!/usr/bin/env python3
"""Entry point: run a full C. elegans connectome emulation episode.

Usage::

    python scripts/run_simulation.py [--steps N] [--seed S] [--data-dir PATH]

Produces:
    - Episode summary printed to stdout.
    - Spike raster, trajectory, and connectome graph plots in ``experiments/results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C. elegans connectome emulation episode."
    )
    parser.add_argument("--steps", type=int, default=None, help="Override sim_steps")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override data directory path"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import os
    if args.seed is not None:
        os.environ["CELEGANS_SEED"] = str(args.seed)
    if args.data_dir is not None:
        os.environ["CELEGANS_DATA_DIR"] = args.data_dir

    from celegans.config import load_config
    from celegans.connectome import load_connectome
    from celegans.simulation import build_simulation_runner
    from celegans.utils.logging import get_logger
    from celegans.utils.reproducibility import set_all_seeds
    from celegans.visualization import (
        plot_connectome_graph,
        plot_spike_raster,
        plot_trajectory,
    )

    logger = get_logger("run_simulation", log_file=_PROJECT_ROOT / "simulation.log")
    cfg = load_config()
    set_all_seeds(cfg.seed)

    data_dir = cfg.resolved_data_dir(_PROJECT_ROOT)
    results_dir = cfg.resolved_results_dir(_PROJECT_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading connectome from %s", data_dir)
    graph = load_connectome(data_dir)

    summary = graph.summary()
    print("\n── Connectome Summary ──────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print("────────────────────────────────────────────────────\n")

    sim_steps = args.steps or cfg.sim_steps
    runner = build_simulation_runner(
        graph=graph,
        sim_steps=sim_steps,
        results_dir=results_dir,
        seed=cfg.seed,
        project_root=_PROJECT_ROOT,
    )

    logger.info("Starting simulation: %d steps", sim_steps)
    result = runner.run_episode()

    print("\n── Episode Result ───────────────────────────────────")
    for k, v in result.summary().items():
        print(f"  {k:30s}: {v}")
    print("────────────────────────────────────────────────────\n")

    # Save visualisations
    logger.info("Saving visualisations...")

    plot_spike_raster(
        result.spike_history,
        node_names=graph.node_names,
        sensory_indices=graph.get_sensory_indices(),
        interneuron_indices=graph.get_interneuron_indices(),
        motor_indices=graph.get_motor_indices(),
        output_path=results_dir / "spike_raster.png",
        project_root=_PROJECT_ROOT,
    )

    # Use centre-ish of trajectory as proxy for food position
    food_pos_proxy = result.trajectory[-1] + 20
    plot_trajectory(
        result.trajectory,
        food_position=food_pos_proxy,
        output_path=results_dir / "trajectory.png",
        project_root=_PROJECT_ROOT,
    )

    plot_connectome_graph(
        graph,
        output_path=results_dir / "connectome_graph.png",
        project_root=_PROJECT_ROOT,
    )

    print(f"Plots saved to: {results_dir}")
    print("Simulation complete.")


if __name__ == "__main__":
    main()
