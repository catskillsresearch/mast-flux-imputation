#!/usr/bin/env python3
"""Predictions for the viewer's side-by-side comparison (all outputs gitignored).

* ``outputs/compare/<split>_e4.npz`` (+ ``.json``, ``.score.json``): the E4 ensemble (FAIR-MAST-only
  calibration; psi from the three E4 members, q95 / betaN from their scalar heads);
* audits ``outputs/audit/compare_<split>_e4`` and, if missing, ``<split>_hybrid_shipped`` for the
  shipped Challenge-2 hybrid (``outputs/<split>_hybrid.npz``), so the viewer can show per-frame
  boundary, point and shape metrics for both.

Existing files are kept (rerun is a no-op).

    .venv/bin/python scripts/make_compare_preds.py demo holdout
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    from experiments.evaluate import audit, predict_split
    out = REPO / "outputs" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    members = [Path(f) for f in json.loads((REPO / "outputs/experiments/E4/members.json").read_text())["files"]]
    py = sys.executable
    for split in sys.argv[1:] or ["demo"]:
        npz = out / f"{split}_e4.npz"
        if not npz.with_suffix(".json").is_file():
            predict_split(split, npz, members=members, scalars="heads")
        if not npz.with_suffix(".score.json").is_file():
            subprocess.run([py, "-m", "pipeline.score", str(npz)], cwd=REPO, check=True)
        audit(npz, f"compare_{split}_e4")
        ship = REPO / "outputs" / f"{split}_hybrid.npz"
        if ship.is_file():
            audit(ship, f"{split}_hybrid_shipped")
        print(split, "ok", npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
