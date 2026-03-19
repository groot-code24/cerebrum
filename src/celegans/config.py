"""Configuration module for C. elegans Connectome Emulator.

All settings are read from environment variables prefixed with CELEGANS_
or from an optional .env file. Uses stdlib dataclasses — no pydantic required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env(key: str, default: str) -> str:
    return os.environ.get(f"CELEGANS_{key.upper()}", default)


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_list_float(key: str, default: List[float]) -> List[float]:
    raw = os.environ.get(f"CELEGANS_{key.upper()}")
    if raw is None:
        return list(default)
    return [float(v.strip()) for v in raw.split(",")]


def _env_list_int(key: str, default: List[int]) -> List[int]:
    raw = os.environ.get(f"CELEGANS_{key.upper()}")
    if raw is None:
        return list(default)
    return [int(v.strip()) for v in raw.split(",")]


def _env_list_str(key: str, default: List[str]) -> List[str]:
    raw = os.environ.get(f"CELEGANS_{key.upper()}")
    if raw is None:
        return list(default)
    return [v.strip() for v in raw.split(",")]


def _load_dotenv(path: Path) -> None:
    """Load key=value pairs from .env file into os.environ (non-overwriting)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip matching outer quote pair only
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            if key not in os.environ:
                os.environ[key] = val


@dataclass
class CelegansConfig:
    """All configurable hyperparameters for the emulator."""

    # Neuron model
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 1.0
    reset_potential: float = 0.0
    dt: float = 0.1

    # GNN architecture
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1
    gnn_aggr: str = "mean"

    # Simulation
    sim_steps: int = 1000
    food_gradient_strength: float = 1.0
    body_segments: int = 10
    physics_substeps: int = 5

    # Ablation
    ablation_fractions: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5]
    )
    ablation_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    specific_ablation_neurons: List[str] = field(
        default_factory=lambda: ["AVB", "AWC", "ASE", "AIY", "AIZ"]
    )

    # Reproducibility
    seed: int = 42

    # Paths
    data_dir: str = "data/raw"
    results_dir: str = "experiments/results"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.tau_mem <= 0:
            raise ValueError(f"tau_mem must be > 0, got {self.tau_mem}")
        if self.tau_syn <= 0:
            raise ValueError(f"tau_syn must be > 0, got {self.tau_syn}")
        if not (0 < self.dt < 1):
            raise ValueError(f"dt must be in (0, 1), got {self.dt}")
        if self.threshold <= self.reset_potential:
            raise ValueError(
                f"threshold ({self.threshold}) must be > reset_potential ({self.reset_potential})"
            )
        if not (0.0 <= self.gnn_dropout < 1.0):
            raise ValueError(f"gnn_dropout must be in [0, 1), got {self.gnn_dropout}")
        if self.gnn_aggr not in {"mean", "sum", "max"}:
            raise ValueError(f"gnn_aggr must be mean/sum/max, got {self.gnn_aggr!r}")
        for f in self.ablation_fractions:
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"ablation fraction must be in [0,1], got {f}")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log_level: {self.log_level!r}")

    def resolved_data_dir(self, project_root: Path) -> Path:
        p = (project_root / self.data_dir).resolve()
        _validate_within_root(p, project_root)
        return p

    def resolved_results_dir(self, project_root: Path) -> Path:
        p = (project_root / self.results_dir).resolve()
        _validate_within_root(p, project_root)
        return p


def _validate_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {path} is outside project root {root}"
        )


def load_config(project_root: Path | None = None) -> CelegansConfig:
    """Load config from .env (if present) then environment variables."""
    root = project_root or Path.cwd()
    _load_dotenv(root / ".env")
    return CelegansConfig(
        tau_mem=_env_float("tau_mem", 20.0),
        tau_syn=_env_float("tau_syn", 5.0),
        threshold=_env_float("threshold", 1.0),
        reset_potential=_env_float("reset_potential", 0.0),
        dt=_env_float("dt", 0.1),
        gnn_hidden_dim=_env_int("gnn_hidden_dim", 64),
        gnn_num_layers=_env_int("gnn_num_layers", 3),
        gnn_dropout=_env_float("gnn_dropout", 0.1),
        gnn_aggr=_env("gnn_aggr", "mean"),
        sim_steps=_env_int("sim_steps", 1000),
        food_gradient_strength=_env_float("food_gradient_strength", 1.0),
        body_segments=_env_int("body_segments", 10),
        physics_substeps=_env_int("physics_substeps", 5),
        ablation_fractions=_env_list_float("ablation_fractions", [0.0, 0.1, 0.25, 0.5]),
        ablation_seeds=_env_list_int("ablation_seeds", [42, 123, 456]),
        specific_ablation_neurons=_env_list_str(
            "specific_ablation_neurons", ["AVB", "AWC", "ASE", "AIY", "AIZ"]
        ),
        seed=_env_int("seed", 42),
        data_dir=_env("data_dir", "data/raw"),
        results_dir=_env("results_dir", "experiments/results"),
        log_level=_env("log_level", "INFO"),
    )
