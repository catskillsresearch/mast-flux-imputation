#!/usr/bin/env python3
"""Impute MAST poloidal flux psi(R,Z), q95 and betaN without magnetic-probe / flux-loop inputs.

psi is the mean of the selected five-member ensemble (``select.json`` of the training run); q95 and
betaN come from the members' scalar heads.

Output: ``<out>.npz`` in the template layout (``shot_XXXX_psirz`` (T,65,65), ``shot_XXXX_q95`` (T,),
``shot_XXXX_betaN`` (T,), XXXX = index in the split order) plus ``<out>.json`` with the shot keys,
exclusions, member files and timing.

    python -m pipeline.predict --split holdout
    python -m pipeline.predict --split demo --out outputs/demo_model.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.datasets import SPLITS  # noqa: E402
from pipeline.paths import assert_not_on_lab, output_dir  # noqa: E402


def main() -> int:
    from experiments.common import EXP
    from experiments.evaluate import predict_split

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=SPLITS, default="holdout")
    ap.add_argument("--select", type=Path, default=EXP / "E6" / "select.json",
                    help="selection record of the training run (its 'files' are the members)")
    ap.add_argument("--out", type=Path, default=None, help="default <outputs>/<split>_model.npz")
    a = ap.parse_args()
    members = [Path(f) for f in json.loads(a.select.read_text())["files"]]
    out = assert_not_on_lab(a.out or output_dir() / f"{a.split}_model.npz")
    predict_split(a.split, out, members=members, scalars="heads")
    print(f"[predict] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
