"""Per-frame EFIT route targets for every windowed training frame (from the cached EFIT psi).

For each frame of data.npz: psi at the O-point (psi_ax) and on the LCFS (psi_b), the LCFS interior
mask on the 65x65 grid, O-point and upper/lower X-point positions (NaN when absent). Extractors are
the audit's (scorer LCFS, scorer magnetic axis, viewer saddle search). Resumable per shot.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.common import COMMON, LAB_CACHE, IP_LO, assert_safe
from models.m9.features import IP_COL

PER = COMMON / "route_shots"


def _one(key: str) -> str:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    dest = PER / f"{key}.npz"
    if dest.is_file():
        return key
    from matplotlib.path import Path as MPath
    from scipy.interpolate import RegularGridInterpolator
    from pipeline import audit as A
    from models.m9 import train as T
    X, Y = T.load_shot(LAB_CACHE, key)
    n = min(len(X), len(Y))
    m = np.abs(X[:n, IP_COL].astype(np.float64)) >= IP_LO
    Y = Y[:n][m].astype(np.float64)
    gz, gr = np.meshgrid(A.Z, A.R, indexing="ij")
    pts = np.column_stack([gr.ravel(), gz.ravel()])
    nT = len(Y)
    psi_ax = np.full(nT, np.nan); psi_b = np.full(nT, np.nan)
    opt = np.full((nT, 2), np.nan); xu = np.full((nT, 2), np.nan); xl = np.full((nT, 2), np.nan)
    inside = np.zeros((nT, 65 * 65), bool)
    for k in range(nT):
        psi = Y[k].reshape(65, 65)
        f = RegularGridInterpolator((A.Z, A.R), psi, bounds_error=False, fill_value=np.nan)
        try:
            c = A.lcfs(psi)
            o = A.opoint(psi)
            xp = A.xpoints(psi)
        except Exception:  # noqa: BLE001
            continue
        opt[k] = o
        psi_ax[k] = float(f([[o[1], o[0]]])[0])
        if xp["upper"] is not None:
            xu[k] = xp["upper"]
        if xp["lower"] is not None:
            xl[k] = xp["lower"]
        if c is not None:
            psi_b[k] = float(np.nanmean(f(np.column_stack([c[:, 1], c[:, 0]]))))
            inside[k] = MPath(c).contains_points(pts)
    tmp = dest.with_name(dest.stem + ".part.npz")
    np.savez(tmp, psi_ax=psi_ax, psi_b=psi_b, opt=opt, xu=xu, xl=xl, inside=np.packbits(inside, axis=1))
    tmp.replace(dest)
    return key


def main(workers: int = 8) -> dict:
    t0 = time.time()
    assert_safe(PER).mkdir(parents=True, exist_ok=True)
    d = np.load(COMMON / "data.npz")
    keys = [str(k) for k in d["keys"]]
    ids = d["ids"]
    todo = [k for k in keys if not (PER / f"{k}.npz").is_file()]
    print(f"[route] {len(keys)} shots, {len(todo)} to do, workers={workers}", flush=True)
    if todo:
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            for i, k in enumerate(ex.map(_one, todo, chunksize=4)):
                if (i + 1) % 200 == 0:
                    print(f"  [route] {i + 1}/{len(todo)} {round(time.time() - t0)} s", flush=True)
    parts = {n: [] for n in ("psi_ax", "psi_b", "opt", "xu", "xl", "inside")}
    for s, k in enumerate(keys):
        z = np.load(PER / f"{k}.npz")
        if len(z["psi_ax"]) != int((ids == s).sum()):
            raise RuntimeError(f"{k}: route frames {len(z['psi_ax'])} != data frames {(ids == s).sum()}")
        for n in parts:
            parts[n].append(z[n])
    out = {n: np.concatenate(v) for n, v in parts.items()}
    ok = np.isfinite(out["psi_ax"]) & np.isfinite(out["psi_b"]) & (out["psi_ax"] > out["psi_b"])
    np.savez(COMMON / "route.npz", ok=ok, **out)
    info = {"frames": int(len(ok)), "ok_frac": float(ok.mean()), "seconds": round(time.time() - t0, 1)}
    print("[route]", info, flush=True)
    return info


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
