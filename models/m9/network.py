"""M9 trunk + temporal features (ported from the catskills_fec M9 trunk module).

Only the pieces the M9 checkpoints use are kept: ``augment_temporal`` (mode
"smoothdiff"), ``_clip_features`` and the ``dualctx`` trunk ``CoeffDualContext``. Numerics are
unchanged. The five scalar heads (q95, betaN, R_axis, Z_axis, li) exist in the checkpoint's
state dict for architectural compatibility but were never trained by the M9 recipe (its loss is
on the PCA coefficients only) and are never read.
"""
from __future__ import annotations

import numpy as np
import torch


def augment_temporal(X: np.ndarray, shot_ids: np.ndarray, window: int, mode: str = "smoothdiff") -> np.ndarray:
    """``(T, F) -> (T, 3F)``: raw frame, centred moving average, smoothed central difference.

    Applied per shot with clamped edges. ``smoothdiff`` differences the moving average
    ``half = window // 2`` frames apart: d_t = (a_{t+h} - a_{t-h}) / (2h).
    """
    X = np.asarray(X, dtype=np.float32)
    if window <= 1:
        return X
    half = int(window) // 2
    out = np.empty((X.shape[0], X.shape[1] * 3), dtype=np.float32)
    out[:, : X.shape[1]] = X
    for s in np.unique(shot_ids):
        sl = np.flatnonzero(shot_ids == s)
        if sl.size == 0:
            continue
        blk = X[sl]
        T = blk.shape[0]
        idx = np.arange(T)
        # Centred moving average with clamped edges.
        acc = np.zeros_like(blk)
        for d in range(-half, half + 1):
            acc += blk[np.clip(idx + d, 0, T - 1)]
        acc /= float(2 * half + 1)
        if mode == "smoothdiff":
            h = max(1, half)
            dif = (acc[np.clip(idx + h, 0, T - 1)] - acc[np.clip(idx - h, 0, T - 1)]) / (2.0 * h)
        else:
            dif = 0.5 * (blk[np.clip(idx + 1, 0, T - 1)] - blk[np.clip(idx - 1, 0, T - 1)])
        out[sl, X.shape[1] : 2 * X.shape[1]] = acc
        out[sl, 2 * X.shape[1] :] = dif
    return out


def _clip_features(Xs: np.ndarray, clip: float, *, label: str = "") -> np.ndarray:
    """Clamp standardized features to +-``clip`` sigma. ``clip <= 0`` disables it."""
    if clip is None or clip <= 0:
        return Xs
    return np.clip(Xs, -clip, clip, out=Xs)


class _ResBlock(torch.nn.Module):
    """Linear - act - dropout - Linear, added back to its own input."""

    def __init__(self, width: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class CoeffDualContext(torch.nn.Module):
    """Two encoders -- present frame and temporal context -- fused by a zero-initialised gate::

        h = blocks(relu(stem(x[:, :F])))          # present frame, F = 308
        c = ctx_blocks(relu(ctx_stem(x[:, F:])))  # [moving average | smoothed difference], 616
        h = h + gate * ctx_proj(c)
        z = coeff_head(h)                         # 60 PCA coefficients
    """

    def __init__(self, n_in: int, n_pca: int, n_split: int, width: int = 1024, blocks: int = 8,
                 ctx_width: int = 512, ctx_blocks: int = 3, dropout: float = 0.05,
                 activation: str = "relu"):
        super().__init__()
        if not 0 < n_split < n_in:
            raise ValueError(f"n_split {n_split} must lie inside (0, n_in={n_in})")
        self.n_split = int(n_split)
        self.width, self.blocks_n = int(width), int(blocks)
        self.ctx_width, self.ctx_blocks_n = int(ctx_width), int(ctx_blocks)
        self.dropout_p, self.activation_name = float(dropout), activation

        self.stem = torch.nn.Linear(self.n_split, self.width)
        self.act = torch.nn.ReLU()
        self.blocks = torch.nn.Sequential(
            *[_ResBlock(self.width, float(dropout), activation) for _ in range(self.blocks_n)]
        )
        self.ctx_stem = torch.nn.Linear(n_in - self.n_split, self.ctx_width)
        self.ctx_blocks = torch.nn.Sequential(
            *[_ResBlock(self.ctx_width, float(dropout), activation) for _ in range(self.ctx_blocks_n)]
        )
        self.ctx_proj = torch.nn.Linear(self.ctx_width, self.width)
        self.gate = torch.nn.Parameter(torch.zeros(1))

        self.coeff_head = torch.nn.Linear(self.width, n_pca)
        # Scalar heads (trained by experiments/e6.py; untrained in the reference M9 trio).
        self.q95_head = torch.nn.Linear(self.width, 1)
        self.beta_head = torch.nn.Linear(self.width, 1)
        self.r_axis_head = torch.nn.Linear(self.width, 1)
        self.z_axis_head = torch.nn.Linear(self.width, 1)
        self.li_head = torch.nn.Linear(self.width, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.act(self.stem(x[:, : self.n_split])))
        c = self.ctx_blocks(self.act(self.ctx_stem(x[:, self.n_split:])))
        return h + self.gate * self.ctx_proj(c)

    def forward(self, x: torch.Tensor):
        h = self.embed(x)
        return (
            self.coeff_head(h),
            self.q95_head(h).squeeze(-1),
            self.beta_head(h).squeeze(-1),
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


# Hyper-parameters of the M9 trunk (catskills_fec M9 training script).
TRUNK = dict(width=512, blocks=4, ctx_width=512, ctx_blocks=3, dropout=0.05, activation="relu")


def build_dualctx(n_in: int, n_pca: int, n_split: int) -> CoeffDualContext:
    return CoeffDualContext(n_in, n_pca, n_split, **TRUNK)


def reconstruct(Z: np.ndarray, mean: np.ndarray, comp: np.ndarray, grid: int = 65) -> np.ndarray:
    """Coefficients ``(n, k)`` -> grids ``(n, 65, 65)``:  psi = mean + Z @ C."""
    return (np.asarray(Z, dtype=np.float32) @ comp + mean).reshape(-1, grid, grid)
