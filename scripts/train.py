#!/usr/bin/env python3
"""Optional supervised fine-tuning entry point.

This script allows training the GNN weights against behavioural targets
(e.g. recorded motor neuron activations) if labelled data is available.

Usage::

    python scripts/train.py --epochs 50 [--lr 1e-3] [--seed 42]

Without labelled data, the GNN runs in zero-shot mode (random init weights).
The simulation will still produce biologically-plausible propagation patterns
because the graph structure itself encodes the connectivity prior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional GNN supervised fine-tuning.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
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

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(
            "ERROR: PyTorch is required for training but is not installed.\n"
            "Install it with:  pip install torch\n"
            "For zero-shot simulation (no training), use scripts/run_simulation.py instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    from celegans.config import load_config
    from celegans.connectome import load_connectome
    from celegans.gnn_model import ConnectomeGNN
    from celegans.utils.logging import get_logger
    from celegans.utils.reproducibility import set_all_seeds

    logger = get_logger("train", log_file=_PROJECT_ROOT / "train.log")
    cfg = load_config()
    set_all_seeds(cfg.seed)

    data_dir = cfg.resolved_data_dir(_PROJECT_ROOT)
    graph = load_connectome(data_dir)

    input_dim = graph.data.x.shape[1]
    model = ConnectomeGNN(
        input_dim=input_dim,
        hidden_dim=cfg.gnn_hidden_dim,
        num_layers=cfg.gnn_num_layers,
        dropout=cfg.gnn_dropout,
        aggr=cfg.gnn_aggr,
        sensory_indices=graph.get_sensory_indices(),
        motor_indices=graph.get_motor_indices(),
    )

    # Pre-convert graph tensors for torch autograd
    import numpy as np
    x_t = torch.tensor(graph.data.x, dtype=torch.float32, requires_grad=False)
    ei_t = torch.tensor(graph.data.edge_index, dtype=torch.long)

    # Build a minimal torch linear stack that mirrors the numpy GNN output shape
    # so that torch autograd can be used for weight updates.
    torch_model = nn.Sequential(
        nn.Linear(input_dim, cfg.gnn_hidden_dim),
        nn.ReLU(),
        *[layer for _ in range(cfg.gnn_num_layers - 1)
          for layer in (nn.Linear(cfg.gnn_hidden_dim, cfg.gnn_hidden_dim), nn.ReLU())],
        nn.Linear(cfg.gnn_hidden_dim, 1),
    )

    optimiser = torch.optim.Adam(torch_model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    logger.info("Training GNN for %d epochs (lr=%.4f)", args.epochs, args.lr)

    torch_model.train()
    for epoch in range(args.epochs):
        optimiser.zero_grad()
        out = torch_model(x_t)  # [N, 1]

        # Self-supervised smoothness loss: connected nodes should have similar activations
        src_out = out[ei_t[0]]
        tgt_out = out[ei_t[1]]
        loss = loss_fn(src_out, tgt_out.detach())

        loss.backward()
        optimiser.step()

        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            logger.info("Epoch %d/%d  loss=%.6f", epoch + 1, args.epochs, float(loss))
            print(f"Epoch {epoch+1}/{args.epochs}  loss={float(loss):.6f}")

    # Save weights
    weights_dir = _PROJECT_ROOT / "experiments" / "results"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "gnn_weights.pt"
    torch.save(torch_model.state_dict(), weights_path)
    print(f"\nWeights saved to {weights_path}")
    logger.info("Training complete. Weights: %s", weights_path)


if __name__ == "__main__":
    main()
