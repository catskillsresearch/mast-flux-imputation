"""Evaluation splits.

``demo``        the 3 labelled FEC starter-kit MAST shots (28348/28350/28351), FEC-native rows.
``holdout``     the M9 sealed-80 FAIR-MAST shots (seed 42; ``data/fair_mast/holdout_seed42.json``)
                re-cut to the FEC protocol:
                  * EFIT clock cut to *diverted* slices (any finite efm X-point), exactly the rule
                    that reproduces the FEC frame selection on the demo shots; magnetics and
                    Thomson clocks left whole (as the FEC organisers did);
                  * GT psi = ingest ``efit_psirz`` on those slices (bit-identical to FEC psi on
                    the demo frames); GT q95 / betaN = efm ``q_95`` / ``betan``;
                  * edge Thomson overlaid from efm-sidecar ``aye`` rows when the ingest parquet
                    has none (the lab ingest predates an edge-clock fix);
                  * shots with no diverted slice are excluded (reported).
``public_test`` the 1,206 unlabelled FEC ``mast_public_test`` shots (inputs only, no scoring).

Every model sees ``inputs_only(row)``: all ``efit_*`` label columns except ``efit_times`` and
the grid are removed before prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from models.io import cut_efit_frames, psi_stack, row_from_parquet
from pipeline.paths import DATA, fair_mast_dir, fair_mast_labels_dir, fec_demo_dir, fec_public_test_dir

SPLITS = ("demo", "holdout", "public_test", "archval100", "select30", "osftrain200")
EXP_SPLITS_JSON = DATA / "fair_mast" / "experiment_splits_seed42.json"
INPUT_EFIT = ("efit_times", "efit_grid_R", "efit_grid_Z")
HOLDOUT_JSON = DATA / "fair_mast" / "holdout_seed42.json"


@dataclass
class Shot:
    key: str
    path: str
    row: dict                      # inputs only
    gt: dict | None = None         # {"psi": (T,65,65), "q95": (T,), "betaN": (T,)}
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(len(np.asarray(self.row["efit_times"]).reshape(-1)))


def inputs_only(row: dict) -> dict:
    return {k: v for k, v in row.items() if not (k.startswith("efit_") and k not in INPUT_EFIT)}


def _f64(v, T=None):
    a = np.asarray(v if v is not None else np.full(T or 0, np.nan), dtype=np.float64).reshape(-1)
    return a


def holdout_keys() -> list[str]:
    return list(json.loads(HOLDOUT_JSON.read_text())["sealed"])


def split_keys(split: str) -> list[str]:
    if split == "demo":
        return sorted(p.stem for p in fec_demo_dir().glob("mast_shot_*.parquet"))
    if split == "holdout":
        return holdout_keys()
    if split in ("archval100", "select30", "osftrain200"):
        return list(json.loads(EXP_SPLITS_JSON.read_text())[split])
    if split == "public_test":
        return sorted(p.stem for p in fec_public_test_dir().glob("mast_shot_*.parquet"))
    raise ValueError(f"unknown split {split!r}; have {SPLITS}")


def load_demo(key: str) -> Shot:
    p = fec_demo_dir() / f"{key}.parquet"
    row = row_from_parquet(p)
    T = len(row["efit_times"])
    gt = {"psi": psi_stack(row), "q95": _f64(row.get("efit_q95"), T), "betaN": _f64(row.get("efit_beta_n"), T)}
    return Shot(key, str(p), inputs_only(row), gt, {"source": "FEC starter kit (labelled demo)"})


def load_holdout(key: str) -> Shot | None:
    """FEC-protocol re-cut of one sealed FAIR-MAST shot; ``None`` if it has no diverted slice."""
    p = fair_mast_dir() / f"{key}.parquet"
    lab_p = fair_mast_labels_dir() / f"{key}.npz"
    if not lab_p.is_file():
        raise FileNotFoundError(f"{lab_p} missing; run data/fair_mast/fetch_efit_labels.py")
    lab = dict(np.load(lab_p))
    row = row_from_parquet(p)
    t_ing = np.asarray(row["efit_times"], dtype=np.float32).reshape(-1)
    if t_ing.shape != lab["time_ms"].shape or not np.array_equal(t_ing, lab["time_ms"]):
        raise RuntimeError(f"{key}: ingest efit_times != efm time; label sidecar is for another clock")
    keep = np.asarray(lab["diverted"], dtype=bool)
    meta = {"source": "FAIR-MAST ingest + efm sidecar", "n_efit_all": int(len(t_ing)),
            "n_diverted": int(keep.sum()), "edge_overlay": False}
    if not keep.any():
        return None
    row = cut_efit_frames(row, keep)
    edge_t = row.get("thomson_edge_times")
    if (edge_t is None or len(np.asarray(edge_t)) == 0) and len(lab.get("edge_times_ms", [])) >= 2:
        row["thomson_edge_times"] = lab["edge_times_ms"].astype(np.float64)
        row["thomson_edge_Te"] = lab["edge_te"].astype(np.float64)
        row["thomson_edge_ne"] = lab["edge_ne"].astype(np.float64)
        if len(lab.get("edge_r", [])):
            row["thomson_edge_spatial"] = lab["edge_r"].astype(np.float64)
        meta["edge_overlay"] = True
    gt = {"psi": psi_stack(row), "q95": lab["q95"][keep].astype(np.float64),
          "betaN": lab["beta_n"][keep].astype(np.float64)}
    return Shot(key, str(p), inputs_only(row), gt, meta)


def load_public_test(key: str) -> Shot:
    p = fec_public_test_dir() / f"{key}.parquet"
    return Shot(key, str(p), inputs_only(row_from_parquet(p)), None, {"source": "FEC mast_public_test"})


def load_shot(split: str, key: str) -> Shot | None:
    if split in ("archval100", "select30", "osftrain200"):  # FAIR-MAST archive shots, FEC protocol cut
        return load_holdout(key)
    return {"demo": load_demo, "holdout": load_holdout, "public_test": load_public_test}[split](key)


def load_split(split: str, limit: int = 0) -> tuple[list[Shot], list[str]]:
    """(usable shots, excluded keys)."""
    keys = split_keys(split)
    if limit:
        keys = keys[:limit]
    shots, excluded = [], []
    for k in keys:
        s = load_shot(split, k)
        if s is None:
            excluded.append(k)
        else:
            shots.append(s)
    return shots, excluded
