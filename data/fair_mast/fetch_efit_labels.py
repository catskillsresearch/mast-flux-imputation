#!/usr/bin/env python3
"""Fetch the EFIT++ scalar labels the FAIR-MAST ingest does not carry.

The ingest parquets (``ingest.py``) hold ``efit_psirz`` but not ``efit_q95`` / ``efit_beta_n``,
and they keep *every* EFIT time slice (ramp-up, limited and ramp-down included), whereas the
Fusion Equilibrium Challenge keeps only diverted slices. This script reads, per shot, the
FAIR-MAST ``efm`` group and writes ``labels/mast_shot_<N>.npz`` with

    time_ms      EFIT clock in ms (float32, identical to the ingest ``efit_times``)
    q95          efm ``q_95``
    beta_n       efm ``betan``
    xpoint{1,2}_{rc,zc}
    diverted     any(isfinite(xpoint*_rc))  -- the FEC frame rule
    edge_times_ms, edge_te, edge_ne, edge_r
                 edge Thomson (``aye``) rows with any finite T_e -- the FEC edge layout --
                 and the time-mean chord radius (as ``ingest.py`` derives ``thomson_core_R``).
                 The lab ingest parquets predate an edge-clock fix and carry no edge Thomson;
                 ``pipeline/datasets.py`` overlays these when the ingest edge columns are empty.

The diverted rule reproduces the FEC frame selection exactly on the 3 labelled starter-kit
shots (36 / 42 / 37 frames); ``q_95`` and ``betan`` match their ``efit_q95`` / ``efit_beta_n``
exactly and the edge Thomson matches their ``thomson_edge_*`` exactly; see
``tests/test_labels.py`` (run after fetching the 3 demo shots' labels).

    python data/fair_mast/fetch_efit_labels.py --keys-json data/fair_mast/holdout_seed42.json
    python data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.paths import assert_not_on_lab, fair_mast_labels_dir  # noqa: E402

ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast/level1/shots"
MS = 1000.0
VARS = ("q_95", "betan", "xpoint1_rc", "xpoint1_zc", "xpoint2_rc", "xpoint2_zc", "li", "magnetic_axis_r",
        "magnetic_axis_z")


def fetch_one(fs, shot: int) -> dict:
    import s3fs
    import xarray as xr

    efm = xr.open_zarr(s3fs.S3Map(f"{BUCKET}/{shot}.zarr/efm", s3=fs))
    out = {"time_ms": (np.asarray(efm["time"], np.float64) * MS).astype(np.float32)}
    for v in VARS:
        a = np.asarray(efm[v], np.float64) if v in efm else np.full(len(out["time_ms"]), np.nan)
        out[{"q_95": "q95", "betan": "beta_n"}.get(v, v)] = a
    xr_ = np.vstack([out["xpoint1_rc"], out["xpoint2_rc"]])
    out["diverted"] = np.isfinite(xr_).any(axis=0)
    try:
        aye = xr.open_zarr(s3fs.S3Map(f"{BUCKET}/{shot}.zarr/aye", s3=fs))
        te = np.asarray(aye["te"], np.float64)
        ne = np.asarray(aye["ne"], np.float64)
        tt = np.asarray(aye["time"], np.float64) * MS
        keep = np.isfinite(te).any(axis=1)
        out["edge_times_ms"] = tt[keep].astype(np.float32)
        out["edge_te"] = te[keep].astype(np.float32)
        out["edge_ne"] = ne[keep].astype(np.float32)
        r = np.asarray(aye["r"], np.float64)
        with np.errstate(all="ignore"):
            out["edge_r"] = (np.nanmean(r, axis=0) if r.ndim == 2 else r).astype(np.float32)
    except Exception:  # noqa: BLE001  (edge Thomson is absent on early shots)
        out["edge_times_ms"] = np.zeros(0, np.float32)
        out["edge_te"] = np.zeros((0, 16), np.float32)
        out["edge_ne"] = np.zeros((0, 16), np.float32)
        out["edge_r"] = np.zeros(0, np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="*", default=[])
    ap.add_argument("--keys-json", type=Path, action="append", default=[],
                    help="json with a 'sealed' or 'keys' list of mast_shot_<N> keys")
    ap.add_argument("--out", type=Path, default=fair_mast_labels_dir())
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    shots = list(a.shots)
    for j in a.keys_json:
        d = json.loads(j.read_text())
        for k in d.get("sealed", []) + d.get("keys", []) + d.get("demo_keys", []):
            shots.append(int(str(k).rsplit("_", 1)[-1]))
    shots = sorted(set(shots))
    out = assert_not_on_lab(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import s3fs

    fs = s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT)

    def job(shot):
        dest = out / f"mast_shot_{shot}.npz"
        if dest.is_file():
            return shot, "cached"
        try:
            rec = fetch_one(fs, shot)
        except Exception as e:  # noqa: BLE001
            return shot, f"FAILED {type(e).__name__}: {e}"
        np.savez(dest, **rec)
        return shot, f"{len(rec['time_ms'])} slices, {int(rec['diverted'].sum())} diverted"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, shots))
    bad = [r for r in res if r[1].startswith("FAILED")]
    for s, msg in res:
        print(f"  [{s}] {msg}")
    print(f"[fetch_efit_labels] {len(res) - len(bad)}/{len(res)} ok -> {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
