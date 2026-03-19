
"""Entry point: run full ablation experiment suite.

Usage::

    python scripts/run_ablation.py [--seed S] [--data-dir PATH]

Produces:
    - ``experiments/results/ablation_table.md``
    - ``experiments/results/ablation_all_results.json``
    - Per-condition JSON files
    - Grouped bar chart PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiment suite.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import os
    if args.seed is not None:
        os.environ["CELEGANS_SEED"] = str(args.seed)
    if args.data_dir is not None:
        os.environ["CELEGANS_DATA_DIR"] = args.data_dir

    from celegans.ablation import AblationExperiment
    from celegans.config import load_config
    from celegans.connectome import load_connectome
    from celegans.simulation import build_simulation_runner
    from celegans.utils.logging import get_logger
    from celegans.utils.reproducibility import set_all_seeds
    from celegans.visualization import plot_ablation_results

    logger = get_logger("run_ablation", log_file=_PROJECT_ROOT / "ablation.log")
    cfg = load_config()
    set_all_seeds(cfg.seed)

    data_dir = cfg.resolved_data_dir(_PROJECT_ROOT)
    results_dir = cfg.resolved_results_dir(_PROJECT_ROOT)

    logger.info("Loading connectome...")
    graph = load_connectome(data_dir)

    runner = build_simulation_runner(
        graph=graph,
        sim_steps=cfg.sim_steps,
        results_dir=results_dir,
        seed=cfg.seed,
        project_root=_PROJECT_ROOT,
    )

    experiment = AblationExperiment(
        base_runner=runner,
        ablation_fractions=cfg.ablation_fractions,
        ablation_seeds=cfg.ablation_seeds,
        specific_neurons=cfg.specific_ablation_neurons,
        results_dir=results_dir,
        project_root=_PROJECT_ROOT,
        episode_seed=cfg.seed,
    )

    print("\nRunning full ablation suite...")
    all_results = experiment.run_full_ablation_suite()

    print(f"\n── Ablation Suite Complete ({len(all_results)} conditions) ──")
    for r in all_results[:10]:  # Print first 10
        neurons = ",".join(r.ablated_neurons) if r.ablated_neurons else f"rnd{r.ablation_fraction:.0%}"
        print(
            f"  {neurons:25s}  loco={r.locomotion_score:.3f}"
            f"  chem={r.chemotaxis_score:.3f}"
            f"  deg={r.behavioral_degradation_pct:.1f}%"
        )
    if len(all_results) > 10:
        print(f"  ... and {len(all_results) - 10} more (see ablation_table.md)")

    # Save bar chart
    plot_ablation_results(
        all_results,
        output_path=results_dir / "ablation_results.png",
        project_root=_PROJECT_ROOT,
    )

    print(f"\nResults saved to: {results_dir}")
    print("Ablation complete.")


if __name__ == "__main__":
    main()
