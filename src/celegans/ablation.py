"""Neuron and synapse ablation experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from celegans.connectome import ConnectomeGraph
from celegans.simulation import EpisodeResult, SimulationRunner, build_simulation_runner
from celegans.utils.io import atomic_write_json, atomic_write_text, validate_path
from celegans.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Outcome of a single ablation experiment."""

    ablated_neurons: List[str]
    ablation_fraction: float
    ablation_seed: int
    locomotion_score: float
    chemotaxis_score: float
    mean_spike_rate_change: float
    behavioral_degradation_pct: float
    food_reached: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def markdown_row(self) -> str:
        neurons = ", ".join(self.ablated_neurons) if self.ablated_neurons else "—"
        return (
            f"| {neurons} "
            f"| {self.ablation_fraction:.0%} "
            f"| {self.locomotion_score:.3f} "
            f"| {self.chemotaxis_score:.3f} "
            f"| {self.mean_spike_rate_change:+.3f} "
            f"| {self.behavioral_degradation_pct:.1f}% "
            f"| {'✓' if self.food_reached else '✗'} |"
        )


class AblationExperiment:
    """Structured ablation experiment suite.

    Args:
        base_runner: Baseline SimulationRunner.
        ablation_fractions: Synapse fractions to test.
        ablation_seeds: RNG seeds for random ablation replicates.
        specific_neurons: Named neurons for targeted silencing.
        results_dir: Output directory.
        project_root: Project root for path safety.
        episode_seed: Seed for each simulation episode.
    """

    def __init__(
        self,
        base_runner: SimulationRunner,
        ablation_fractions: Optional[List[float]] = None,
        ablation_seeds: Optional[List[int]] = None,
        specific_neurons: Optional[List[str]] = None,
        results_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
        episode_seed: int = 42,
    ) -> None:
        self.base_runner = base_runner
        self.ablation_fractions = ablation_fractions or [0.0, 0.1, 0.25, 0.5]
        self.ablation_seeds = ablation_seeds or [42, 123, 456]
        self.specific_neurons = specific_neurons or ["AVB", "AWC", "ASE", "AIY", "AIZ"]
        self.episode_seed = episode_seed
        self._project_root = project_root or Path.cwd()

        if results_dir is not None:
            self.results_dir = validate_path(results_dir, self._project_root)
        else:
            self.results_dir = (self._project_root / "experiments" / "results").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._baseline: Optional[EpisodeResult] = None

    def get_baseline(self) -> EpisodeResult:
        if self._baseline is None:
            logger.info("Running baseline episode...")
            self._baseline = self.base_runner.run_episode(episode_seed=self.episode_seed)
        return self._baseline

    def run_specific_ablation(self, neuron_names: List[str]) -> AblationResult:
        baseline = self.get_baseline()
        all_nodes = set(self.base_runner.graph.node_names)
        valid: List[str] = []
        for n in neuron_names:
            if n in all_nodes:
                valid.append(n)
            else:
                # Expand prefix: "AVB" → ["AVBL", "AVBR"], "AWC" → ["AWCL", "AWCR"]
                expanded = [node for node in all_nodes if node.startswith(n)]
                if expanded:
                    logger.info(
                        "Neuron %r not found directly — expanding to prefix matches: %s",
                        n, expanded,
                    )
                    valid.extend(expanded)
        if not valid:
            logger.warning("None of %s found in graph — returning baseline scores.", neuron_names)
            return _make_result(neuron_names, 0.0, -1, baseline, baseline)
        ablated_graph = self.base_runner.graph.ablate_neurons(valid)
        ablated_runner = _clone_runner_with_graph(self.base_runner, ablated_graph)
        ep = ablated_runner.run_episode(episode_seed=self.episode_seed)
        return _make_result(valid, 0.0, -1, ep, baseline)

    def run_random_ablation(self, fraction: float, seed: int) -> AblationResult:
        baseline = self.get_baseline()
        ablated_graph = self.base_runner.graph.ablate_random_synapses(fraction, seed)
        ablated_runner = _clone_runner_with_graph(self.base_runner, ablated_graph)
        ep = ablated_runner.run_episode(episode_seed=self.episode_seed)
        return _make_result([], fraction, seed, ep, baseline)

    def run_full_ablation_suite(self) -> List[AblationResult]:
        all_results: List[AblationResult] = []
        for neuron in self.specific_neurons:
            r = self.run_specific_ablation([neuron])
            all_results.append(r)
            self._save_single_result(r, f"ablation_neuron_{neuron}")
        for frac in self.ablation_fractions:
            for seed in self.ablation_seeds:
                r = self.run_random_ablation(frac, seed)
                all_results.append(r)
                self._save_single_result(r, f"ablation_random_frac{int(frac*100)}_seed{seed}")
        self._save_markdown_table(all_results)
        self._save_all_results_json(all_results)
        logger.info("Full ablation suite: %d conditions", len(all_results))
        return all_results

    def _save_single_result(self, result: AblationResult, label: str) -> None:
        atomic_write_json(self.results_dir / f"{label}.json", result.to_dict())

    def _save_all_results_json(self, results: List[AblationResult]) -> None:
        atomic_write_json(
            self.results_dir / "ablation_all_results.json",
            [r.to_dict() for r in results],
        )

    def _save_markdown_table(self, results: List[AblationResult]) -> None:
        header = (
            "# Ablation Results\n\n"
            "| Ablated Neurons | Synapse Fraction | Locomotion Score "
            "| Chemotaxis Score | ΔSpike Rate | Degradation | Food Reached |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        rows = "\n".join(r.markdown_row() for r in results)
        atomic_write_text(self.results_dir / "ablation_table.md", header + rows + "\n")


def _make_result(
    ablated_neurons: List[str],
    ablation_fraction: float,
    ablation_seed: int,
    ep: EpisodeResult,
    baseline: EpisodeResult,
) -> AblationResult:
    base_disp = max(baseline.total_displacement, 1e-6)
    loco_score = min(ep.total_displacement / base_disp, 2.0)
    base_motor = baseline.mean_spike_rate_by_type.get("motor", 1e-6)
    ep_motor = ep.mean_spike_rate_by_type.get("motor", 0.0)
    # Chemotaxis score: 1.0 if outcome matches baseline, 0.0 if worse, up to 2.0 if better.
    # When baseline did not reach food, score baseline as 0.5 to avoid division by near-zero.
    baseline_chem = 1.0 if baseline.food_reached else 0.5
    ep_chem = 1.0 if ep.food_reached else 0.0
    chemotaxis_score = min(ep_chem / baseline_chem, 2.0)
    mean_spike_change = (ep_motor - base_motor) / max(base_motor, 1e-9)
    loco_deg = max(0.0, 1.0 - loco_score) * 100.0
    chem_deg = max(0.0, 1.0 - chemotaxis_score) * 100.0
    degradation = (loco_deg + chem_deg) / 2.0
    return AblationResult(
        ablated_neurons=ablated_neurons,
        ablation_fraction=ablation_fraction,
        ablation_seed=ablation_seed,
        locomotion_score=round(loco_score, 4),
        chemotaxis_score=round(chemotaxis_score, 4),
        mean_spike_rate_change=round(mean_spike_change, 6),
        behavioral_degradation_pct=round(degradation, 2),
        food_reached=ep.food_reached,
    )


def _clone_runner_with_graph(
    runner: SimulationRunner, new_graph: ConnectomeGraph
) -> SimulationRunner:
    from celegans.config import load_config
    from celegans.environment import WormEnv
    from celegans.gnn_model import ConnectomeGNN
    from celegans.lif_neurons import LIFNeuronBank

    cfg = load_config()
    n = new_graph.data.num_nodes
    sensory_idx = new_graph.get_sensory_indices()
    motor_idx = new_graph.get_motor_indices()
    interneuron_idx = new_graph.get_interneuron_indices()
    input_dim = new_graph.data.x.shape[1]

    model = ConnectomeGNN(
        input_dim=input_dim, hidden_dim=cfg.gnn_hidden_dim,
        num_layers=cfg.gnn_num_layers, dropout=cfg.gnn_dropout,
        aggr=cfg.gnn_aggr, sensory_indices=sensory_idx, motor_indices=motor_idx,
    )
    lif_bank = LIFNeuronBank(
        n_total=n, sensory_indices=sensory_idx,
        interneuron_indices=interneuron_idx, motor_indices=motor_idx,
        tau_sensory=10.0, tau_interneuron=cfg.tau_mem, tau_motor=15.0,
        dt=cfg.dt, threshold=cfg.threshold, reset_potential=cfg.reset_potential,
    )
    n_motor = max(len(motor_idx), 1)
    env = WormEnv(
        n_neurons=n, body_segments=cfg.body_segments,
        physics_substeps=cfg.physics_substeps,
        food_gradient_strength=cfg.food_gradient_strength,
        dorsal_motor_indices=motor_idx[:n_motor // 2],
        ventral_motor_indices=motor_idx[n_motor // 2:],
        num_motor_neurons=n_motor, seed=runner.seed,
    )
    return SimulationRunner(
        graph=new_graph, model=model, lif_bank=lif_bank, env=env,
        sim_steps=runner.sim_steps, results_dir=runner.results_dir,
        seed=runner.seed, project_root=runner._project_root,
    )
