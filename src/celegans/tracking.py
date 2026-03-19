"""Experiment tracking for ablation and simulation runs.

Provides a clean, context-manager-based API that writes structured JSON logs
locally and optionally syncs to MLflow when available.

Usage
-----
::

    from celegans.tracking import ExperimentTracker

    with ExperimentTracker("ablation_AVAL", results_dir=Path("experiments/results")) as tracker:
        tracker.log_params({"neuron": "AVAL", "seed": 42, "dt": 0.1})
        # ... run simulation ...
        tracker.log_metrics({"locomotion_score": 0.72, "chemotaxis_index": 0.61})
        tracker.log_artifact(raster_plot_path)

All runs are written to::

    experiments/results/runs/<run_name>_<timestamp>/
        params.json
        metrics.json
        artifacts/   (symlinks or copies)

MLflow
------
If ``mlflow`` is installed, every :class:`ExperimentTracker` instance
automatically starts an MLflow run in the ``celegans-emulator`` experiment.
Set ``MLFLOW_TRACKING_URI`` env var to point to a remote server.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from celegans.utils.io import atomic_write_json
from celegans.utils.logging import get_logger

logger = get_logger(__name__)

# Optional MLflow
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """File-based experiment tracker with optional MLflow backend.

    Parameters
    ----------
    run_name : str
        Human-readable name for this run.
    results_dir : Path
        Root directory for experiment output.
    experiment_name : str
        MLflow experiment name (also used as sub-directory grouping).
    tags : dict, optional
        Key-value metadata tags.
    use_mlflow : bool
        Explicitly enable/disable MLflow (default: auto-detect).
    """

    def __init__(
        self,
        run_name: str,
        results_dir: Path = Path("experiments/results"),
        experiment_name: str = "celegans-emulator",
        tags: Optional[Dict[str, str]] = None,
        use_mlflow: Optional[bool] = None,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags or {}

        ts = int(time.time())
        self.run_dir = Path(results_dir) / "runs" / f"{run_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)

        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, List[float]] = {}
        self._step: int = 0
        self._start_time: float = time.perf_counter()
        self._closed: bool = False

        # MLflow
        self._mlflow_run = None
        _use_mlflow = use_mlflow if use_mlflow is not None else _MLFLOW_AVAILABLE
        if _use_mlflow and _MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment_name)
                self._mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)
                logger.info("MLflow run started: %s", self._mlflow_run.info.run_id)
            except Exception as exc:
                logger.warning("MLflow unavailable: %s", exc)
                self._mlflow_run = None

        logger.info("ExperimentTracker: %s → %s", run_name, self.run_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (call once before run starts)."""
        self._params.update(params)
        atomic_write_json(self.run_dir / "params.json", self._params)
        if self._mlflow_run:
            try:
                # MLflow only accepts str values for params
                mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception as exc:
                logger.debug("MLflow log_params failed: %s", exc)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log scalar metrics at a given step."""
        s = step if step is not None else self._step
        for k, v in metrics.items():
            if k not in self._metrics:
                self._metrics[k] = []
            self._metrics[k].append({"step": s, "value": float(v)})

        # Write full metrics history atomically
        atomic_write_json(self.run_dir / "metrics.json", self._metrics)

        if self._mlflow_run:
            try:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=s)
            except Exception as exc:
                logger.debug("MLflow log_metrics failed: %s", exc)

        self._step += 1

    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """Copy a file into this run's artifact directory."""
        path = Path(path)
        if not path.exists():
            logger.warning("Artifact not found: %s", path)
            return
        dest_name = name or path.name
        dest = self.run_dir / "artifacts" / dest_name
        shutil.copy2(path, dest)
        if self._mlflow_run:
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:
                logger.debug("MLflow log_artifact failed: %s", exc)

    def log_figure(self, fig, name: str) -> None:
        """Save a matplotlib figure as a PNG artifact."""
        try:
            import matplotlib.pyplot as plt
            dest = self.run_dir / "artifacts" / name
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if self._mlflow_run:
                try:
                    mlflow.log_artifact(str(dest))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("log_figure failed: %s", exc)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value
        if self._mlflow_run:
            try:
                mlflow.set_tag(key, value)
            except Exception:
                pass

    def close(self) -> None:
        """Finalise run: write summary, close MLflow run."""
        if self._closed:
            return
        elapsed = time.perf_counter() - self._start_time
        summary = {
            "run_name": self.run_name,
            "elapsed_seconds": round(elapsed, 3),
            "params": self._params,
            "final_metrics": {
                k: v[-1]["value"] for k, v in self._metrics.items()
            },
            "tags": self.tags,
        }
        atomic_write_json(self.run_dir / "run_summary.json", summary)

        if self._mlflow_run:
            try:
                mlflow.log_metric("elapsed_seconds", elapsed)
                mlflow.end_run()
            except Exception as exc:
                logger.debug("MLflow end_run failed: %s", exc)

        logger.info(
            "Run complete: %s  (%.1fs)  → %s",
            self.run_name, elapsed, self.run_dir,
        )
        self._closed = True

    # Context manager
    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.set_tag("status", "FAILED")
            self.set_tag("error", str(exc_val)[:200])
        else:
            self.set_tag("status", "FINISHED")
        self.close()

    # ------------------------------------------------------------------
    # Convenience: compare runs
    # ------------------------------------------------------------------

    @staticmethod
    def load_run(run_dir: Path) -> Dict[str, Any]:
        """Load a previously saved run summary."""
        summary_path = Path(run_dir) / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No run_summary.json in {run_dir}")
        with summary_path.open() as f:
            return json.load(f)

    @staticmethod
    def compare_runs(run_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Load and sort multiple runs by a metric for comparison."""
        runs = []
        for d in run_dirs:
            try:
                runs.append(ExperimentTracker.load_run(d))
            except FileNotFoundError:
                pass
        return sorted(runs, key=lambda r: r.get("elapsed_seconds", 0))
