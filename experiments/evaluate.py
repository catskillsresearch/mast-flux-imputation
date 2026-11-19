"""Predict a split with an experiment ensemble (or the reference M9) and audit it.

Scalar sources: ``heads`` (M9 scalar heads), ``osf_e5`` (OSF read-outs through the E5 maps fitted on
FAIR-MAST training shots), ``osf_raw`` (OSF read-outs unchanged), ``osf_shipped`` (reference c*, s*
map; baseline only), ``const`` (q95 = 3.5, betaN = 1.5; for psi-only checkpoint selection).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.common import EXP, REPO, assert_safe

E5_MAPS = EXP / "E5" / "maps.json"


def osf_for(split: str, keys: list[str], workers: int = 8) -> dict:
    from pipeline.predict import run_osf
    return run_osf(split, keys, REPO / "outputs" / "cache" / "osf" / split, workers)


def e5_apply(q, b):
    m = json.loads(E5_MAPS.read_text())
    q = np.clip(m["a_q"] * np.asarray(q, float) + m["b_q"], 2.0, 25.0)
    b = np.clip(m["a_b"] * np.asarray(b, float) + m["b_b"], 0.2, 5.0)
    return q, b


def predict_split(split: str, out_npz: Path, *, members=None, shipped: bool = False, scalars: str = "const",
                  psi_from: str = "m9", workers: int = 8) -> Path:
    from pipeline.datasets import load_shot, load_split
    out_npz = assert_safe(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shots, excluded = load_split(split)
    keys = [s.key for s in shots]
    del shots
    need_osf = scalars.startswith("osf") or psi_from == "osf"
    osf = osf_for(split, keys, workers) if need_osf else None
    ens = None
    if psi_from == "m9":
        if shipped:
            from models.m9 import M9Ensemble
            from pipeline.paths import m9_weights_dir
            ens = M9Ensemble(m9_weights_dir())
        else:
            from experiments.m9x import Ensemble
            ens = Ensemble(members)
    arrays = {}
    nf = 0
    for i, k in enumerate(keys):
        heads = None
        if psi_from == "osf":
            psi = osf[k]["psirz"]
        else:
            row = load_shot(split, k).row
            if shipped:
                psi = ens.predict(row)
            else:
                psi, heads = ens.predict(row)
        T = psi.shape[0]
        if scalars == "heads":
            q, b = heads[:, 0], heads[:, 1]
        elif scalars == "osf_e5":
            q, b = e5_apply(osf[k]["q95"], osf[k]["betaN"])
        elif scalars == "osf_raw":
            q, b = osf[k]["q95"], osf[k]["betaN"]
        elif scalars == "osf_shipped":
            from models.osf import apply_final_scalars
            q, b = apply_final_scalars(osf[k]["q95"], osf[k]["betaN"])
        else:
            q, b = np.full(T, 3.5), np.full(T, 1.5)
        arrays[f"shot_{i:04d}_psirz"] = np.asarray(psi, np.float32)
        arrays[f"shot_{i:04d}_q95"] = np.asarray(q, np.float32)
        arrays[f"shot_{i:04d}_betaN"] = np.asarray(b, np.float32)
        nf += T
    np.savez_compressed(out_npz, **arrays)
    meta = {"split": split, "mode": f"{psi_from}+{scalars}", "keys": keys, "excluded_no_diverted_frames": excluded,
            "members": [str(m) for m in (members or [])], "shipped": shipped, "n_frames": nf,
            "seconds": round(time.time() - t0, 1)}
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return out_npz


def audit(pred: Path, tag: str, workers: int = 8) -> dict:
    from pipeline import audit as A
    out = REPO / "outputs" / "audit" / tag
    if not (out / "summary.json").is_file():
        A.run(pred, tag, None, workers)
    return json.loads((out / "summary.json").read_text())


def compare_table(old_tag: str, new_tag: str) -> str:
    """Markdown old-vs-new table (pipeline.audit compare), returned as text."""
    import contextlib
    import io
    from pipeline import audit as A
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.compare(REPO / "outputs" / "audit" / old_tag, REPO / "outputs" / "audit" / new_tag)
    return buf.getvalue()
