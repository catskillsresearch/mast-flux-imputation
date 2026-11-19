#!/usr/bin/env python3
"""Ingest MAST shots from the public FAIR-MAST archive into the FEC parquet schema.

Port of catskills_fec catskills/scripts/mast_ingest.py (numerics unchanged). FAIR-MAST
(https://mastapp.site, anonymous S3 at s3.echo.stfc.ac.uk, CC BY-SA 4.0) publishes the
MAST M5-M9 archive; it is the upstream source the Fusion Equilibrium Challenge itself was
built from.

Per shot it reads the Zarr groups efm (EFIT++: psirz on a doubled 129-column R grid, of
which the 65 finite columns are kept), amc (coil and plasma currents, resampled onto the
FEC 30,000-sample 0.2 ms magnetics grid), ayc (core Thomson) and aye (edge Thomson),
and writes one parquet row in the mast_public_test column layout *plus* the label
efit_psirz. The output grid (``efit_grid_R``, ``efit_grid_Z``) is the shot's own efm ``r`` / ``z``.
Everything in a row comes from FAIR-MAST; no other data source is read.

Caveat (paraphrasing the original docstring): of ~4,566 archive shots carrying both EFIT and
core Thomson the challenge curated 2,416, so part of this pool is the challenge's hidden test
folds and cannot be excluded. A model trained on it is not a zero-shot MAST result.

    python data/fair_mast/ingest.py --pool data/fair_mast/mast_pool3.txt --shots 4 \
        --out data/fair_mast/parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.paths import assert_not_on_lab, fair_mast_dir  # noqa: E402

ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast/level1/shots"

COILS = ["efps", "sol", "tf"] + [f"p{i}{ab}" for i in range(2, 7) for ab in ("l", "u")]
# The archive keeps every clock in seconds; the challenge republishes them in ms, and puts
# magnetics on one canonical grid regardless of the shot's own acquisition length.
MS = 1000.0
MAG_GRID = np.linspace(-2000.0, 3999.7998, 30000)


def static_row() -> dict:
    """No external per-machine tables: the grid is read per shot from efm (see ``one_shot``)."""
    return {}


def thomson_times_to_ms(tt, efit_times_ms) -> np.ndarray | None:
    """Archive-native seconds -> ms. Drop clocks that miss the EFIT window."""
    tt = np.asarray(tt, np.float64)
    if tt.size < 2 or not np.isfinite(tt).any():
        return None
    efit = np.asarray(efit_times_ms, np.float64)
    if efit.size == 0 or not np.isfinite(efit).any():
        return None
    lo = float(np.nanmin(efit)) - 50.0
    hi = float(np.nanmax(efit)) + 50.0

    def n_olap(t: np.ndarray) -> int:
        return int(np.sum(np.isfinite(t) & (t >= lo) & (t <= hi)))

    t_s = tt * MS
    if n_olap(t_s) >= 2:
        return t_s
    if n_olap(tt) >= 2:
        return tt
    return None


def _grp(fs, shot: int, name: str):
    import xarray as xr
    import s3fs
    return xr.open_zarr(s3fs.S3Map(f"{BUCKET}/{shot}.zarr/{name}", s3=fs))


def _ragged(a: np.ndarray) -> np.ndarray:
    """(T, N) -> object array of T float64 rows, the layout the challenge parquet uses."""
    out = np.empty(len(a), dtype=object)
    for i, row in enumerate(a):
        out[i] = np.asarray(row, np.float64)
    return out


def _cell(v):
    """Arrow wants nested lists, not n-d numpy, for the (T, 65, 65) and ragged columns."""
    if isinstance(v, str) or np.isscalar(v):
        return v
    a = np.asarray(v, dtype=object) if getattr(v, "dtype", None) == object else np.asarray(v)
    return a.tolist() if a.ndim > 1 or a.dtype == object else a


def one_shot(fs, shot: int, static: dict) -> dict | None:
    efm = _grp(fs, shot, "efm")
    psi = np.asarray(efm["psirz"])                       # (T, 65, 129)
    keep = np.isfinite(psi[0]).all(0)                    # the 65 real R columns
    if keep.sum() != 65 or psi.shape[1] != 65:
        print(f"  [{shot}] unexpected psirz {psi.shape}, {int(keep.sum())} real cols")
        return None
    psi = psi[:, :, keep].astype(np.float32)             # (T, 65, 65)
    r = np.asarray(efm["profile_r"])[keep]
    if not np.allclose(r, np.asarray(efm["r"])):
        print(f"  [{shot}] R grid mismatch after de-interleave")
        return None

    row: dict = {"source": "MAST", **static}
    row["efit_grid_R"] = np.asarray(efm["r"], np.float64)
    row["efit_grid_Z"] = np.asarray(efm["z"], np.float64)
    row["efit_times"] = (np.asarray(efm["time"], np.float64) * MS).astype(np.float32)
    row["efit_psirz"] = psi
    if len(psi) != len(row["efit_times"]):
        print(f"  [{shot}] psi/time length mismatch")
        return None

    amc = _grp(fs, shot, "amc")
    tm = np.asarray(amc["time"], np.float64) * MS
    row["magnetics_time"] = MAG_GRID.astype(np.float32)
    for c in COILS + ["plasma"]:
        key = "plasma_current" if c == "plasma" else f"{c}_current"
        if key not in amc:
            print(f"  [{shot}] missing {key}")
            return None
        y = np.asarray(amc[key], np.float64)
        row[f"magnetics_{key}"] = np.interp(MAG_GRID, tm, y, left=np.nan, right=np.nan)

    for grp, tag, rkey, spatial in (("ayc", "core", "radius", "thomson_core_R"),
                                    ("aye", "edge", "r", "thomson_edge_spatial")):
        try:
            ts = _grp(fs, shot, grp)
        except Exception:
            # Edge Thomson is absent for the early M5 shots; core is required.
            if grp == "aye":
                row["thomson_edge_times"] = np.zeros(0, np.float64)
                row["thomson_edge_Te"] = np.empty(0, dtype=object)
                row["thomson_edge_ne"] = np.empty(0, dtype=object)
                continue
            print(f"  [{shot}] no {grp}")
            return None
        te = np.asarray(ts["te"], np.float64) if "te" in ts else None
        ne = np.asarray(ts["ne"], np.float64) if "ne" in ts else None
        if te is None or ne is None or te.ndim != 2:
            print(f"  [{shot}] {grp} te/ne unusable")
            return None
        tt = np.asarray(ts["time"], np.float64)
        # Archive clocks are seconds; challenge EFIT is ms. The old edge guard
        # ``max < 10`` dropped valid 60 s aye windows (demo Te matches *1000).
        if tag == "edge":
            times_ms = thomson_times_to_ms(tt, row["efit_times"])
            if times_ms is None:
                row["thomson_edge_times"] = np.zeros(0, np.float64)
                row["thomson_edge_Te"] = np.empty(0, dtype=object)
                row["thomson_edge_ne"] = np.empty(0, dtype=object)
                continue
            row["thomson_edge_times"] = times_ms.astype(np.float32)
        else:
            row[f"thomson_{tag}_times"] = tt * MS
        row[f"thomson_{tag}_Te"] = _ragged(te)
        row[f"thomson_{tag}_ne"] = _ragged(ne)
        rad = np.asarray(ts[rkey], np.float64) if rkey in ts else None
        if rad is not None:
            row[spatial] = rad.mean(0) if rad.ndim == 2 else rad
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(ROOT / "data" / "fair_mast" / "mast_pool3.txt"),
                    help="shot ids with both efm and ayc (from the FAIR-MAST source index)")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--keys", type=int, nargs="*", default=[],
                    help="explicit shot numbers (overrides --pool / --start / --shots)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(fair_mast_dir()))
    args = ap.parse_args()

    import s3fs
    fs = s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT)
    pool = [int(x) for x in Path(args.pool).read_text().split()]
    todo = args.keys or pool[args.start:args.start + args.shots]
    out = assert_not_on_lab(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    static = static_row()
    print(f"[mast_ingest] {len(todo)} shots -> {out}", flush=True)

    ok = 0
    for i, shot in enumerate(todo):
        dest = out / f"mast_shot_{shot}.parquet"
        if dest.is_file():
            ok += 1
            continue
        try:
            row = one_shot(fs, shot, static)
        except Exception as e:
            print(f"  [{shot}] FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if row is None:
            continue
        pd.DataFrame([{k: _cell(v) for k, v in row.items()}]).to_parquet(dest)
        ok += 1
        T = len(row["efit_times"])
        print(f"  [{shot}] {T:4d} frames  psi {row['efit_psirz'].shape}  "
              f"core TS {len(row['thomson_core_times'])}  -> {dest.name}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(todo)}  ok={ok}", flush=True)
    print(f"[mast_ingest] wrote {ok}/{len(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
