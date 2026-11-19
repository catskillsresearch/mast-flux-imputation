#!/usr/bin/env python3
"""Model predictions for the viewer (all outputs git-ignored).

For each split, writes ``outputs/compare/<split>_e6.npz`` (+ ``.json``, ``.score.json``): the
five-member ensemble chosen by ``experiments/driver_e6.py`` (``outputs/experiments/E6/select.json``),
psi from the members and q95 / betaN from their scalar heads; and the observable audit
``outputs/audit/compare_<split>_e6`` with per-frame boundary, point and shape metrics.

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
    sel = REPO / "outputs/experiments/E6/select.json"
    members = [Path(f) for f in json.loads(sel.read_text())["files"]]
    py = sys.executable
    for split in sys.argv[1:] or ["demo"]:
        npz = out / f"{split}_e6.npz"
        if not npz.with_suffix(".json").is_file():
            predict_split(split, npz, members=members, scalars="heads")
        if not npz.with_suffix(".score.json").is_file():
            subprocess.run([py, "-m", "pipeline.score", str(npz)], cwd=REPO, check=True)
        audit(npz, f"compare_{split}_e6")
        print(split, "ok", npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
