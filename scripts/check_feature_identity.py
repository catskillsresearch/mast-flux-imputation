#!/usr/bin/env python3
"""Check that rows built by ``data/fair_mast/ingest.py`` give the same model features as an
existing feature cache.

For each sampled shot the row is downloaded from FAIR-MAST into a git-ignored folder, turned into
the 308-column raw feature matrix X and the flux target Y with ``models.m9.features``, and compared
element by element (NaN == NaN) with the cached X, Y. Nothing is written outside ``--out``.

    python scripts/check_feature_identity.py --n 20 --cache <feature cache dir>
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=True))


def main() -> int:
    from models.io import row_from_parquet
    from models.m9.features import EDGE_START, build_one, shot_from_mast_row
    from pipeline.datasets import inputs_only

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--keys", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "feature_check")
    a = ap.parse_args()
    keys = a.keys
    if not keys:
        allk = sorted(int(p.stem.split("_")[-1]) for p in a.cache.glob("mast_shot_*.npz"))
        keys = sorted(random.Random(a.seed).sample(allk, a.n))
    pq = a.out / "parquet"
    pq.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO / "data/fair_mast/ingest.py"), "--keys", *map(str, keys),
                    "--out", str(pq)], check=True)
    rows, n_ok = [], 0
    for k in keys:
        f = pq / f"mast_shot_{k}.parquet"
        c = a.cache / f"mast_shot_{k}.npz"
        if not f.is_file() or not c.is_file():
            rows.append({"shot": k, "status": "missing", "parquet": f.is_file(), "cache": c.is_file()})
            continue
        row = row_from_parquet(f)
        X, Y = build_one(shot_from_mast_row(inputs_only(row) | {"efit_psirz": row["efit_psirz"]}))
        with np.load(c) as d:
            Xc, Yc = d["X"], d["Y"]
        rec = {"shot": k, "frames": int(len(X)), "X_identical": same(X, Xc), "Y_identical": same(Y, Yc),
               "X_core_identical": same(X[:, :EDGE_START], Xc[:, :EDGE_START]) if X.shape == Xc.shape else False,
               "columns": len(row)}
        if X.shape == Xc.shape and not rec["X_identical"]:
            d = np.abs(np.nan_to_num(X.astype(np.float64)) - np.nan_to_num(Xc.astype(np.float64)))
            rec["X_maxdiff"] = float(d.max())
            rec["X_cols_differ"] = np.flatnonzero(d.max(0) > 0).tolist()[:20]
        n_ok += rec["X_identical"] and rec["Y_identical"]
        rows.append(rec)
    rep = {"cache": str(a.cache), "shots": keys, "identical": int(n_ok), "checked": len(keys), "rows": rows}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for r in rows:
        print(r)
    print(f"[feature-identity] {n_ok}/{len(keys)} shots bit-identical in X and Y")
    return 0 if n_ok == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
