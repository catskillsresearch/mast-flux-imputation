"""Shared paths, constants and status helpers for the E1-E5 experiments."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAB_CACHE = Path(os.environ.get("MFI_FEATURE_CACHE",
                                "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/cache/mast_features_v1"))
EXP = REPO / "outputs" / "experiments"
COMMON = EXP / "common"
SPLITS_JSON = REPO / "data" / "fair_mast" / "experiment_splits_seed42.json"
SHIPPED_IP_LO = 328.7293519015368
SHIPPED_WEIGHTS = REPO / "models" / "m9" / "weights"
MEMBER_SEEDS = (101, 202, 303)
TAGS = {101: "M9_win50_i101", 202: "M9_win50_i202", 303: "M9_win50_i303"}

# Route score (pre-declared, docs/TRAINING_REGIME.md section 5): mean ratio to the reference model.
ROUTE_KEYS = ("rate_jump", "p95_hausdorff_mm", "mean_hausdorff_mm", "mean_area_symdiff_rel",
              "mean_xpt_disp_mm", "mean_opoint_disp_mm", "rate_topology_flip")
FLIP_EPS = 0.005


def route_score(summ: dict, ref: dict) -> float:
    vals = []
    for k in ROUTE_KEYS:
        a, b = float(summ[k]), float(ref[k])
        if k == "rate_topology_flip":
            a, b = a + FLIP_EPS, b + FLIP_EPS
        vals.append(a / b)
    return float(sum(vals) / len(vals))


def assert_safe(p: Path) -> Path:
    p = Path(p).resolve()
    if str(p).startswith("/media/catskills/CRC_FUSION_LAB"):
        raise RuntimeError(f"refusing to write on the lab drive: {p}")
    return p


def status_update(stage: str, **kw) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    f = EXP / "STATUS.json"
    st = json.loads(f.read_text()) if f.is_file() else {"stages": {}}
    rec = st["stages"].get(stage, {})
    rec.update(kw)
    rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    st["stages"][stage] = rec
    st["updated"] = rec["updated"]
    st["pid"] = os.getpid()
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str) + "\n")
    tmp.replace(f)


def stage_done(stage: str) -> bool:
    f = EXP / "STATUS.json"
    if not f.is_file():
        return False
    return json.loads(f.read_text())["stages"].get(stage, {}).get("state") == "done"
