"""Dump the ground truth of a scored split to one npz so an external scorer can re-score it.

    python tests/crosscheck_dump_gt.py outputs/compare/holdout_e6.npz /tmp/holdout_gt.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.datasets import load_shot  # noqa: E402

pred, out = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads(pred.with_suffix(".json").read_text())
arr = {}
for i, k in enumerate(meta["keys"]):
    g = load_shot(meta["split"], k).gt
    arr[f"shot_{i:04d}_psirz"] = g["psi"]
    arr[f"shot_{i:04d}_q95"] = g["q95"]
    arr[f"shot_{i:04d}_betaN"] = g["betaN"]
arr["keys"] = np.array(meta["keys"])
np.savez(out, **arr)
print("wrote", out, len(meta["keys"]), "shots")
