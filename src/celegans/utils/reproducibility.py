"""Seed management and deterministic mode utilities — stdlib + numpy only."""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set seeds for all available RNG sources.

    Covers: Python ``random``, NumPy.
    Also seeds torch and torch_geometric when available.

    Args:
        seed: Non-negative integer seed value.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
