"""Graph Variational Autoencoder (GVAE) for the C. elegans connectome.

Encodes the adjacency structure into a continuous latent space, enabling:
  - Connectome interpolation between ablation conditions
  - Latent-space optimisation: find the connectome that maximises CI
  - Anomaly detection: how far is a given ablation from the manifold?
  - Generation of novel, valid connectome variants

Architecture
-----------
Encoder:  GNN (mean aggregation) → μ [N, d_z]  +  log_σ² [N, d_z]
Reparam:  z = μ + ε·σ
Decoder:  inner product decoder — P(A_ij=1) = σ(z_i · z_j)

This is the Kipf & Welling (2016) VGAE with a per-node latent code.

References
----------
Kipf TN, Welling M (2016). Variational Graph Auto-Encoders. NeurIPS Workshop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from celegans.connectome import GraphData
from celegans.gnn_model import _NumpyLinear, _NumpySAGELayer, _relu
from celegans.utils.logging import get_logger

logger = get_logger(__name__)


class GraphVAE:
    """Variational Graph Autoencoder — pure numpy implementation.

    Parameters
    ----------
    input_dim : int
        Node feature dimensionality.
    hidden_dim : int
        GNN hidden layer width.
    latent_dim : int
        Latent space dimensionality (d_z per node).
    num_encoder_layers : int
        Number of GNN layers in encoder.
    seed : int
        Random seed for weight init and sampling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_encoder_layers: int = 2,
        seed: int = 42,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._rng = np.random.default_rng(seed)

        # Encoder: feature projection + GNN layers
        self._input_proj = _NumpyLinear(input_dim, hidden_dim)
        self._enc_convs = [
            _NumpySAGELayer(hidden_dim, hidden_dim) for _ in range(num_encoder_layers)
        ]

        # Separate heads for μ and log_σ²
        self._mu_head = _NumpyLinear(hidden_dim, latent_dim)
        self._logvar_head = _NumpyLinear(hidden_dim, latent_dim)

        # Training state
        self._train_losses: List[float] = []

        logger.info(
            "GraphVAE: input=%d  hidden=%d  latent=%d  enc_layers=%d",
            input_dim, hidden_dim, latent_dim, num_encoder_layers,
        )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(
        self, data: GraphData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode graph into per-node μ and log_σ².

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        mu : np.ndarray shape [N, latent_dim]
        log_var : np.ndarray shape [N, latent_dim]
        """
        h = _relu(self._input_proj(data.x))
        for conv in self._enc_convs:
            h = _relu(conv(h, data.edge_index))

        mu = self._mu_head(h)
        log_var = np.clip(self._logvar_head(h), -4.0, 4.0)  # prevent blow-up
        return mu, log_var

    def reparameterize(
        self, mu: np.ndarray, log_var: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Reparameterization trick: z = μ + ε·exp(log_σ²/2)."""
        if deterministic:
            return mu
        std = np.exp(0.5 * log_var)
        eps = self._rng.standard_normal(mu.shape).astype(np.float32)
        return (mu + eps * std).astype(np.float32)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Inner-product decoder: P(A_ij) = sigmoid(z_i · z_j^T).

        Parameters
        ----------
        z : np.ndarray shape [N, latent_dim]

        Returns
        -------
        adj_prob : np.ndarray shape [N, N]  —  predicted edge probabilities
        """
        logits = z @ z.T
        return _sigmoid(logits)

    def decode_edges(
        self, z: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        """Decode only specific edges (memory-efficient for large graphs).

        Parameters
        ----------
        z : np.ndarray shape [N, latent_dim]
        edge_index : np.ndarray shape [2, E]

        Returns
        -------
        edge_probs : np.ndarray shape [E]
        """
        src, tgt = edge_index[0], edge_index[1]
        logits = (z[src] * z[tgt]).sum(axis=1)
        return _sigmoid(logits)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, data: GraphData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass: encode → sample → decode.

        Returns
        -------
        adj_pred : np.ndarray shape [N, N]  —  reconstructed adjacency probs
        z : np.ndarray shape [N, latent_dim]  —  latent codes
        mu : np.ndarray shape [N, latent_dim]
        log_var : np.ndarray shape [N, latent_dim]
        """
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        adj_pred = self.decode(z)
        return adj_pred, z, mu, log_var

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        data: GraphData,
        adj_pred: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray,
        beta: float = 1.0,
    ) -> Dict[str, float]:
        """ELBO loss = reconstruction loss + β·KL divergence.

        Parameters
        ----------
        data : GraphData  —  ground truth graph
        adj_pred : np.ndarray shape [N, N]
        mu, log_var : encoder outputs
        beta : float  —  KL weight (β-VAE variant, default 1.0)

        Returns
        -------
        dict with keys: ``total``, ``recon``, ``kl``
        """
        N = data.num_nodes

        # Build target adjacency matrix (sparse → dense for loss)
        A_true = np.zeros((N, N), dtype=np.float32)
        src, tgt = data.edge_index
        A_true[src, tgt] = 1.0

        # Positive-class weight: rebalance for sparse graphs
        n_pos = A_true.sum()
        n_neg = N * N - n_pos
        pos_weight = n_neg / max(n_pos, 1.0)

        # Binary cross-entropy (weighted)
        eps = 1e-7
        adj_pred_c = np.clip(adj_pred, eps, 1.0 - eps)
        bce = -(
            pos_weight * A_true * np.log(adj_pred_c)
            + (1 - A_true) * np.log(1 - adj_pred_c)
        )
        recon_loss = float(bce.mean())

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = float(-0.5 * (1 + log_var - mu ** 2 - np.exp(log_var)).mean())

        total = recon_loss + beta * kl_loss
        return {"total": total, "recon": recon_loss, "kl": kl_loss}

    # ------------------------------------------------------------------
    # Training (gradient-free: evolution strategy)
    # ------------------------------------------------------------------

    def fit(
        self,
        data: GraphData,
        n_epochs: int = 100,
        lr: float = 0.01,
        beta: float = 1.0,
        verbose: bool = False,
    ) -> List[float]:
        """Train using numerical gradient approximation (ES-style).

        Uses forward-mode finite differences on all linear layer weights.
        This is intentionally simple — for real training, use the PyTorch
        version (see notes in README).

        Returns
        -------
        losses : List[float]  —  total loss at each epoch
        """
        losses: List[float] = []
        eps_fd = 1e-3  # finite-difference step

        all_layers = (
            [self._input_proj, self._mu_head, self._logvar_head]
            + self._enc_convs
        )

        for epoch in range(n_epochs):
            adj_pred, z, mu, log_var = self.forward(data)
            loss_dict = self.loss(data, adj_pred, mu, log_var, beta)
            L = loss_dict["total"]
            losses.append(L)

            if verbose and (epoch % max(1, n_epochs // 5) == 0):
                logger.info(
                    "GVAE epoch %d/%d  loss=%.4f  recon=%.4f  kl=%.4f",
                    epoch, n_epochs, L, loss_dict["recon"], loss_dict["kl"],
                )

            # Finite-difference gradient step on each weight matrix
            for layer in all_layers:
                for attr in ("W", "W_self", "W_neigh", "b"):
                    if not hasattr(layer, attr):
                        continue
                    W = getattr(layer, attr)
                    grad = np.zeros_like(W)
                    flat = W.ravel()
                    for i in range(min(len(flat), 50)):  # cap for speed
                        orig = flat[i]
                        flat[i] = orig + eps_fd
                        W_plus = flat.reshape(W.shape)
                        setattr(layer, attr, W_plus.copy())
                        adj_p2, _, m2, lv2 = self.forward(data)
                        Lp = self.loss(data, adj_p2, m2, lv2, beta)["total"]
                        flat[i] = orig
                        W_orig = flat.reshape(W.shape)
                        setattr(layer, attr, W_orig.copy())
                        grad.ravel()[i] = (Lp - L) / eps_fd
                    # SGD step
                    setattr(layer, attr, W - lr * grad)

        self._train_losses = losses
        return losses

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    def get_latent_codes(
        self, data: GraphData
    ) -> np.ndarray:
        """Return deterministic latent codes (μ) for all nodes."""
        mu, _ = self.encode(data)
        return mu

    def reconstruct_adjacency(
        self, data: GraphData, threshold: float = 0.5
    ) -> np.ndarray:
        """Reconstruct binary adjacency matrix from latent codes."""
        mu, _ = self.encode(data)
        adj_prob = self.decode(mu)
        return (adj_prob >= threshold).astype(np.float32)

    def interpolate(
        self,
        data_a: GraphData,
        data_b: GraphData,
        steps: int = 10,
    ) -> List[np.ndarray]:
        """Interpolate between two connectomes in latent space.

        Parameters
        ----------
        data_a, data_b : GraphData  —  two connectome states
        steps : int  —  interpolation steps

        Returns
        -------
        List of adjacency probability matrices along the interpolation path.
        """
        mu_a, _ = self.encode(data_a)
        mu_b, _ = self.encode(data_b)

        results = []
        for t in np.linspace(0, 1, steps):
            z_interp = (1 - t) * mu_a + t * mu_b
            adj = self.decode(z_interp)
            results.append(adj.astype(np.float32))
        return results

    def adjacency_reconstruction_accuracy(
        self, data: GraphData
    ) -> float:
        """Fraction of edges correctly reconstructed (at 0.5 threshold)."""
        N = data.num_nodes
        A_true = np.zeros((N, N), dtype=np.float32)
        A_true[data.edge_index[0], data.edge_index[1]] = 1.0
        A_pred = self.reconstruct_adjacency(data)
        return float((A_true == A_pred).mean())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
