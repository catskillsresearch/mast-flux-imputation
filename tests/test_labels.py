"""Check that fetch_efit_labels.py reproduces the FEC demo labels and frame selection.

    python data/fec/fetch.py --demo
    python data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351
    python tests/test_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.paths import fair_mast_labels_dir, fec_demo_dir  # noqa: E402


def main() -> int:
    bad = 0
    for shot in (28348, 28350, 28351):
        row = pq.read_table(fec_demo_dir() / f"mast_shot_{shot}.parquet").to_pylist()[0]
        lab = np.load(fair_mast_labels_dir() / f"mast_shot_{shot}.npz")
        d = lab["diverted"].astype(bool)
        t_fec = np.asarray(row["efit_times"], np.float64)
        t_lab = lab["time_ms"][d].astype(np.float64)
        checks = {
            "n_frames": d.sum() == len(t_fec),
            "times(1 ms)": len(t_lab) == len(t_fec) and np.max(np.abs(np.round(t_lab) - t_fec)) <= 0.5,
            "q95": np.array_equal(lab["q95"][d].astype(np.float32), np.asarray(row["efit_q95"], np.float32)),
            "betaN": np.array_equal(lab["beta_n"][d].astype(np.float32), np.asarray(row["efit_beta_n"], np.float32)),
            "edge_Te": np.array_equal(np.asarray(lab["edge_te"], np.float32),
                                      np.asarray(row["thomson_edge_Te"], np.float32), equal_nan=True),
        }
        print(shot, int(d.sum()), {k: bool(v) for k, v in checks.items()})
        bad += sum(not v for v in checks.values())
    print("OK" if bad == 0 else f"{bad} FAILED")
    return int(bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
