#!/usr/bin/env python3
"""Find the no-plasma FAIR-MAST shots and their effect on the M9 |I_p| window (metadata only).

Reads, per ingest parquet, only ``efit_times`` and the ``efit_psirz`` column statistics; per
feature-cache npz, only ``X`` (the plasma-current column). Writes
``outputs/compat/ingest_efit_meta.json`` and ``outputs/compat/no_plasma.json`` and prints a summary.
Signature: exactly 200 EFIT slices on the fixed 5 ms clock (reduced ``efm`` product).

    python scripts/scan_no_plasma_shots.py --feature-cache <dir>/mast_features_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.m9 import train as T  # noqa: E402
from models.m9.features import IP_COL  # noqa: E402
from pipeline.paths import assert_not_on_lab, fair_mast_dir, output_dir  # noqa: E402

IP_LO = 328.7293519015368


def main() -> int:
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-cache", type=Path, default=T.DEFAULT_CACHE)
    a = ap.parse_args()
    keep = (ROOT / "data/fair_mast/mast_qc_keep.txt").read_text().split()
    rows = []
    for f in sorted(fair_mast_dir().glob("mast_shot_*.parquet")):
        pf = pq.ParquetFile(f)
        md = pf.metadata.row_group(0)
        lo = hi = None
        for i in range(md.num_columns):
            c = md.column(i)
            if c.path_in_schema.startswith("efit_psirz") and c.statistics is not None and c.statistics.has_min_max:
                lo, hi = float(c.statistics.min), float(c.statistics.max)
        t = np.asarray(pf.read(columns=["efit_times"]).column(0)[0].as_py(), float)
        rows.append({"key": f.stem, "shot": int(f.stem.split("_")[-1]), "T": len(t),
                     "t0": float(t[0]) if len(t) else None, "t1": float(t[-1]) if len(t) else None,
                     "dt": float(np.median(np.diff(t))) if len(t) > 1 else None,
                     "psi_min": lo, "psi_max": hi, "keep": f.stem in set(keep)})
    odd = {r["key"] for r in rows if r["T"] == 200 and r["dt"] is not None and abs(r["dt"] - 5) < 0.01}
    ipmax, nwin = {}, {}
    for k in keep:
        p = a.feature_cache / f"{k}.npz"
        if p.exists():
            with np.load(p) as z:
                ip = np.abs(z["X"][:, IP_COL].astype(float))
            ipmax[k], nwin[k] = float(np.nanmax(ip)), int((ip >= IP_LO).sum())
    odd_keep = [k for k in keep if k in odd]
    res = {"n_ingest": len(rows), "n_no_plasma": len(odd), "n_no_plasma_in_keep": len(odd_keep),
           "no_plasma_max_ip_kA_max": max(ipmax[k] for k in odd_keep if k in ipmax),
           "no_plasma_windowed_frames": sum(nwin.get(k, 0) for k in odd_keep),
           "ip_lo_with": T.fit_ip_lo(a.feature_cache, keep)["ip_lo"],
           "ip_lo_without": T.fit_ip_lo(a.feature_cache, [k for k in keep if k not in odd])["ip_lo"],
           "no_plasma_keys": sorted(odd)}
    out = assert_not_on_lab(output_dir() / "compat")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingest_efit_meta.json").write_text(json.dumps(rows) + "\n")
    (out / "no_plasma.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: v for k, v in res.items() if k != "no_plasma_keys"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
