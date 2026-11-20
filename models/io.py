"""Row I/O shared by the model, the pipeline and the viewer.

``row_from_parquet`` and ``interpolate`` are copied from catskills_fec
(``catskills/warm_feature_cache.py`` and ``src/feq/data.py``) without change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Columns that live on the EFIT clock (cut together when frames are dropped).
EFIT_CLOCK_COLUMNS = (
    "efit_times", "efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis",
    "efit_z_axis", "efit_beta_p", "efit_elongation", "efit_psi_axis", "efit_psi_boundary",
    "efit_geom_axis_r", "efit_geom_axis_z", "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z",
)


def row_from_parquet(path: Path) -> dict:
    """Load the single-shot row from a parquet shard without datasets streaming."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows < 1:
        raise ValueError(f"empty parquet: {path}")
    # One shot per file in this corpus.
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def interpolate(source_t, values, target_t) -> np.ndarray:
    """NaN-aware 1-D interpolation along the first axis.

    Supports `(T,)` and `(T, C)` values. Channels with fewer than two finite
    observations remain NaN.
    """
    x = np.asarray(source_t, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64)
    q = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if y.shape[0] != x.size:
        raise ValueError(f"time/value mismatch: {x.size} vs {y.shape[0]}")
    flat = y.reshape(x.size, -1)
    out = np.full((q.size, flat.shape[1]), np.nan, dtype=np.float64)
    for j in range(flat.shape[1]):
        good = np.isfinite(x) & np.isfinite(flat[:, j])
        if good.sum() >= 2:
            xj, yj = x[good], flat[good, j]
            inside = (q >= xj[0]) & (q <= xj[-1])
            out[inside, j] = np.interp(q[inside], xj, yj)
    return out.reshape((q.size,) + y.shape[1:])


def psi_stack(row: dict, key: str = "efit_psirz") -> np.ndarray | None:
    """``efit_psirz`` as a dense (T, 65, 65) float64 array (rows = Z, cols = R)."""
    if key not in row or row[key] is None:
        return None
    psi = np.asarray(row[key], dtype=np.float64)
    if psi.ndim == 1:
        psi = psi.reshape(-1, 65, 65)
    elif psi.ndim == 2:
        psi = psi.reshape(psi.shape[0], 65, 65)
    return psi


def cut_efit_frames(row: dict, keep: np.ndarray) -> dict:
    """Keep only the EFIT frames ``keep`` (bool or index), exactly as the FEC organizers did
    for the diverted-only re-cut: every column on the EFIT clock is subset, the magnetics and
    Thomson clocks are left whole. ``efit_times`` simply stops being contiguous."""
    out = dict(row)
    t = np.asarray(row["efit_times"], dtype=np.float64).reshape(-1)
    idx = np.flatnonzero(keep) if np.asarray(keep).dtype == bool else np.asarray(keep, dtype=int)
    for col in EFIT_CLOCK_COLUMNS:
        if col not in row or row[col] is None:
            continue
        v = row[col]
        if col == "efit_psirz":
            out[col] = psi_stack(row)[idx]
            continue
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            arr = np.asarray(v, dtype=object)
        if arr.shape[:1] == (t.size,):
            out[col] = arr[idx]
    return out
