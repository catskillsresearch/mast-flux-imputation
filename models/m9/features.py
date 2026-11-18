"""M9 input features (from catskills_fec ``catskills/mast_cache.py`` + ``catskills/features.py``).

Per EFIT frame t the raw feature vector x_t in R^308 is

    x[0:14]    14 magnetics currents linearly interpolated (interp1d, extrapolated) onto efit_times:
               efps, sol, tf, p2l, p2u, p3l, p3u, p4l, p4u, p5l, p5u, p6l, p6u, plasma (I_p last)
    x[14:276]  core Thomson: 131 T_e channels then 131 n_e channels, each channel interpolated in time
               onto efit_times (zero outside the Thomson clock); NO spatial remapping
    x[276:308] edge Thomson: 16 T_e + 16 n_e channels (the shipped M9 zeroes these at inference,
               ``mask_edge=True``)

Code below is copied without numerical change; only imports were made self-contained.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

COILS = ["efps", "sol", "tf"] + [f"p{i}{ab}" for i in range(2, 7) for ab in ("l", "u")]
# Archive magnetics, stable order. plasma last so Ip is a fixed column.
MAG_NAMES = tuple(f"{c}_current" for c in COILS) + ("plasma_current",)
# 14 magnetics + core Te/ne (131x2=262) = 276; + edge Te/ne (16x2=32) = 308.
N_FEATURES = 308
EDGE_START = N_FEATURES - 32
IP_COL = MAG_NAMES.index("plasma_current")


def _interp_masked(times: np.ndarray, values: np.ndarray, efit_times: np.ndarray) -> np.ndarray:
    tt = np.asarray(times, dtype=np.float64)
    yy = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(tt) & np.isfinite(yy)
    if mask.sum() < 2:
        return np.zeros(len(efit_times), dtype=np.float32)
    f = interp1d(tt[mask], yy[mask], kind="linear", fill_value="extrapolate", bounds_error=False)
    return f(efit_times).astype(np.float32)


def interpolate_thomson_to_efit(shot: dict) -> np.ndarray | None:
    efit_times = shot["efit_times"]
    all_features = []
    for system_key in ["thomson_core", "thomson_edge"]:
        if system_key not in shot:
            continue
        system = shot[system_key]
        for measure in ["Te", "ne"]:
            if measure not in system:
                continue
            profiles = system[measure]
            thomson_times = system["times"]
            n_channels = profiles.shape[1] if profiles.ndim == 2 else 1
            for ch in range(n_channels):
                ch_data = profiles[:, ch] if profiles.ndim == 2 else profiles
                mask = np.isfinite(ch_data)
                if mask.sum() < 2:
                    all_features.append(np.zeros(len(efit_times), dtype=np.float32))
                    continue
                f = interp1d(thomson_times[mask], ch_data[mask], kind="linear", fill_value=0.0, bounds_error=False)
                all_features.append(f(efit_times).astype(np.float32))
    if not all_features:
        return None
    return np.column_stack(all_features)


def shot_from_mast_row(row: dict) -> dict:
    """Internal shot dict from an ingest / demo / public-test / private-test MAST parquet row."""
    shot: dict = {"source": "MAST"}
    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)
    if "efit_psirz" in row and row["efit_psirz"] is not None:
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim == 1:
            psi = psi.reshape(-1, 65, 65)
        shot["efit_psirz"] = psi
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    magnetics = {}
    for name in MAG_NAMES:
        col = f"magnetics_{name}"
        if col not in row or row[col] is None:
            continue
        magnetics[name] = {
            "values": np.asarray(row[col], dtype=np.float32),
            "times": mag_t,
        }
    shot["magnetics"] = magnetics
    if row.get("thomson_core_times") is not None and len(np.asarray(row["thomson_core_times"])):
        shot["thomson_core"] = {
            "times": np.asarray(row["thomson_core_times"], dtype=np.float64),
            "Te": np.asarray(row["thomson_core_Te"], dtype=np.float32),
            "ne": np.asarray(row["thomson_core_ne"], dtype=np.float32),
        }
    if row.get("thomson_edge_times") is not None and len(np.asarray(row["thomson_edge_times"])):
        try:
            shot["thomson_edge"] = {
                "times": np.asarray(row["thomson_edge_times"], dtype=np.float64),
                "Te": np.asarray(row["thomson_edge_Te"], dtype=np.float32),
                "ne": np.asarray(row["thomson_edge_ne"], dtype=np.float32),
            }
        except Exception:
            pass
    if row.get("thomson_chord_Z") is not None:
        try:
            shot["thomson_chords"] = {
                "name": np.asarray(row["thomson_chord_name"]).astype(str),
                "Z": np.asarray(row["thomson_chord_Z"], dtype=np.float64),
                "R": np.asarray(row["thomson_chord_R"], dtype=np.float64),
            }
        except Exception:
            pass
    scalars = {}
    for name in ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"):
        if name in row and row[name] is not None:
            scalars[name] = np.asarray(row[name], dtype=np.float64).ravel()
    shot["scalars"] = scalars
    return shot


def magnetics_to_efit(shot: dict) -> np.ndarray:
    """(T, 14) native MAST currents on EFIT times. Missing coil -> zeros."""
    efit_times = shot["efit_times"]
    cols = []
    mag = shot.get("magnetics") or {}
    for name in MAG_NAMES:
        rec = mag.get(name)
        if rec is None:
            cols.append(np.zeros(len(efit_times), dtype=np.float32))
            continue
        cols.append(_interp_masked(rec["times"], rec["values"], efit_times))
    return np.column_stack(cols)


def build_one(shot: dict) -> tuple[np.ndarray, np.ndarray | None]:
    X = magnetics_to_efit(shot)
    th = interpolate_thomson_to_efit(shot)
    if th is not None:
        X = np.hstack([X, th.astype(np.float32)])
    Y = None
    if "efit_psirz" in shot and shot["efit_psirz"] is not None:
        Y = np.asarray(shot["efit_psirz"], dtype=np.float32)
        n = min(len(X), len(Y))
        X = X[:n].astype(np.float32)
        Y = Y[:n]
    else:
        n = len(X)
        X = X.astype(np.float32)
    if X.shape[1] < N_FEATURES:
        X = np.hstack([X, np.zeros((n, N_FEATURES - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]
    return X, Y
