"""Predict a split with a model ensemble and audit it.

Scalar sources: ``heads`` (the model's q95 and betaN heads) or ``const`` (q95 = 3.5, betaN = 1.5;
for psi-only checkpoint selection).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pipeline.paths import output_dir
from experiments.common import EXP, assert_safe

WORKERS = 8


def predict_split(split: str, out_npz: Path, *, members, scalars: str = "heads", workers: int = WORKERS) -> Path:
    from experiments.m9x import Ensemble
    from pipeline.datasets import load_shot, load_split
    out_npz = assert_safe(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shots, excluded = load_split(split)
    keys = [s.key for s in shots]
    del shots
    ens = Ensemble(members)
    arrays = {}
    nf = 0
    for i, k in enumerate(keys):
        psi, heads = ens.predict(load_shot(split, k).row)
        T = psi.shape[0]
        if scalars == "heads":
            q, b = heads[:, 0], heads[:, 1]
        else:
            q, b = np.full(T, 3.5), np.full(T, 1.5)
        arrays[f"shot_{i:04d}_psirz"] = np.asarray(psi, np.float32)
        arrays[f"shot_{i:04d}_q95"] = np.asarray(q, np.float32)
        arrays[f"shot_{i:04d}_betaN"] = np.asarray(b, np.float32)
        nf += T
    np.savez_compressed(out_npz, **arrays)
    meta = {"split": split, "mode": f"m9+{scalars}", "keys": keys, "excluded_no_diverted_frames": excluded,
            "members": [str(m) for m in members], "n_frames": nf, "seconds": round(time.time() - t0, 1)}
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return out_npz


def audit(pred: Path, tag: str, workers: int = WORKERS) -> dict:
    from pipeline import audit as A
    out = output_dir() / "audit" / tag
    if not (out / "summary.json").is_file():
        A.run(pred, tag, None, workers)
    return json.loads((out / "summary.json").read_text())


def pred_audit(split: str, tag: str, **kw) -> dict:
    """Predict (if not already) and audit (if not already); returns the audit summary."""
    summ = output_dir() / "audit" / tag / "summary.json"
    if summ.is_file():
        return json.loads(summ.read_text())
    npz = EXP / "preds" / f"{tag}.npz"
    if not npz.with_suffix(".json").is_file():
        predict_split(split, npz, workers=WORKERS, **kw)
    return audit(npz, tag, WORKERS)


def compare_table(old_tag: str, new_tag: str) -> str:
    """Markdown old-vs-new table (pipeline.audit compare), returned as text."""
    import contextlib
    import io
    from pipeline import audit as A
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.compare(output_dir() / "audit" / old_tag, output_dir() / "audit" / new_tag)
    return buf.getvalue()
