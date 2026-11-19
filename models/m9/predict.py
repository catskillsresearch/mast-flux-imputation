"""M9_win50 ensemble inference (as in catskills_fec ``generate_final_predictions.generate_mast``).

    psi_hat = (1/3) sum_{s in {101,202,303}}  ( mu + C^T f_s( clip( (A(x) - m_s) / sd_s, +-10 ) ) )

with A the per-shot temporal augmentation (window 9, smoothdiff), (m_s, sd_s, mu, C) stored in
each checkpoint, and the edge-Thomson columns 276:308 zeroed before augmentation (mask_edge).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.m9.features import EDGE_START, N_FEATURES, build_one, shot_from_mast_row
from models.m9.network import _clip_features, augment_temporal, build_dualctx, reconstruct

MEMBER_TAGS = ("M9_win50_i101", "M9_win50_i202", "M9_win50_i303")
# sha256 of the reference M9 members.
SHIPPED_SHA256 = {
    "M9_win50_i101": "7660c1d3ad26e00ae088eee3a05264b7fb4582562f732ba95d78b66bb1b2cf2a",
    "M9_win50_i202": "06ba7ef2ff6139884e2fcca9d854abb7b294ae663e60af5882c6b7a02d98dace",
    "M9_win50_i303": "de3b695050acd32ef384811c817f242cae20b4dbc677a675eb828e20f00bee1f",
}


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class M9Ensemble:
    def __init__(self, weights_dir: Path, device: str | None = None, verify_sha: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members = []
        self.meta = []
        for tag in MEMBER_TAGS:
            path = Path(weights_dir) / f"CRC_FEQ_mast_{tag}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing M9 member {path}; set M9_WEIGHTS_DIR")
            sha = _sha256(path)
            if verify_sha and sha != SHIPPED_SHA256[tag]:
                raise RuntimeError(f"{path} sha256 {sha} is not the reference M9 member "
                                   f"(pass verify_sha=False to use a rebuilt checkpoint)")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_dualctx(int(ck["n_features"]), int(ck["n_pca"]), int(ck["n_raw_features"]))
            model.load_state_dict(ck["model_state"])
            model.to(self.device).eval()
            self.members.append((model, ck))
            self.meta.append({"tag": tag, "path": str(path), "sha256": sha,
                              "n_pca": int(ck["n_pca"]), "train_shots": int(ck.get("train_shots", -1)),
                              "seed": int(ck.get("seed", -1)), "init_seed": int(ck.get("init_seed", -1)),
                              "best_epoch": int(ck.get("best_epoch", -1))})

    @torch.no_grad()
    def _infer_z(self, model, xs: torch.Tensor, bs: int = 4096) -> np.ndarray:
        out = []
        for s in range(0, len(xs), bs):
            z = model(xs[s:s + bs].to(self.device, non_blocking=True))[0]
            out.append(z.detach().float().cpu())
        return torch.cat(out).numpy().astype(np.float32)

    def features(self, row: dict) -> np.ndarray:
        X, _ = build_one(shot_from_mast_row(row))
        X = np.array(X, copy=True)
        X[:, EDGE_START:N_FEATURES] = 0.0
        return X

    def predict_members(self, row: dict) -> list[np.ndarray]:
        X = self.features(row)
        ids = np.zeros(len(X), np.int32)
        outs = []
        for model, ck in self.members:
            n_raw = int(ck["n_raw_features"])
            Xw = X if X.shape[1] == n_raw else (
                np.hstack([X, np.zeros((len(X), n_raw - X.shape[1]), np.float32)])
                if X.shape[1] < n_raw else X[:, :n_raw])
            Xa = augment_temporal(Xw, ids, int(ck["temporal_window"]), ck["temporal_mode"])
            Xp = _clip_features(((Xa - ck["feature_mean"]) / ck["feature_std"]).astype(np.float32),
                                float(ck["feature_clip"]))
            z = self._infer_z(model, torch.from_numpy(np.ascontiguousarray(Xp)))
            outs.append(np.asarray(reconstruct(z, ck["pca_mean"], ck["pca_components"]), np.float64))
        return outs

    def predict(self, row: dict) -> np.ndarray:
        """(T, 65, 65) ensemble-mean psi on ``row['efit_times']``."""
        return np.mean(self.predict_members(row), axis=0)
