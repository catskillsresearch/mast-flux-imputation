"""Recreate the M9 sealed-80 holdout and prove it is disjoint from M9 training.

Protocol (catskills_fec ``catskills/run_crc_feq_pinn_mast.py::_split``, unchanged):

    keys   = sorted FAIR-MAST ingest shot keys  (== mast_features_v1 index order)
    keys  -= {28348, 28350, 28351}               (the 3 FEC starter-kit demo shots)
    random.Random(42).shuffle(keys)
    sealed = keys[:80] ;  train_pool = keys[80:]
    train  = train_pool  intersect  mast_qc_keep.txt   (3622 shots, used by every M9 member)

Writes ``data/fair_mast/holdout_seed42.json`` with the list and the checks.

    python data/fair_mast/make_holdout.py [--ingest-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pipeline.paths import fair_mast_dir  # noqa: E402

DEMO_KEYS = ("mast_shot_28348", "mast_shot_28350", "mast_shot_28351")
SEED = 42
N_SEALED = 80


def split(all_keys: list[str], seed: int = SEED):
    keys = [k for k in sorted(all_keys) if k not in DEMO_KEYS]
    random.Random(seed).shuffle(keys)
    return keys[:N_SEALED], keys[N_SEALED:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "holdout_seed42.json")
    a = ap.parse_args()
    d = a.ingest_dir or fair_mast_dir()
    all_keys = sorted(p.stem for p in Path(d).glob("mast_shot_*.parquet"))
    if not all_keys:
        raise SystemExit(f"no mast_shot_*.parquet under {d}")
    sealed, pool = split(all_keys)
    qc = json.loads((HERE / "mast_qc_v1_index.json").read_text())
    keep = set((HERE / "mast_qc_keep.txt").read_text().split())
    quar_train = {q["key"] for q in qc["quarantine"]}
    quar_sealed = {q["key"] for q in qc["quarantine_sealed"]}
    train = [k for k in pool if k in keep]
    checks = {
        "n_ingest_shots": len(all_keys),
        "n_sealed": len(sealed),
        "n_train_pool": len(pool),
        "n_train_qc_keep": len(train),
        "qc_train_set_equals_recomputed_pool": (set(qc["keep"]) | quar_train) == set(pool),
        "qc_keep_equals_keep_txt": set(qc["keep"]) == keep,
        "keep_intersect_sealed": len(keep & set(sealed)),
        "pool_intersect_sealed": len(set(pool) & set(sealed)),
        "demo_intersect_sealed": len(set(DEMO_KEYS) & set(sealed)),
        "quarantine_sealed_subset_of_sealed": quar_sealed <= set(sealed),
        "quarantine_sealed": sorted(quar_sealed),
    }
    ok = (checks["qc_train_set_equals_recomputed_pool"] and checks["qc_keep_equals_keep_txt"]
          and checks["keep_intersect_sealed"] == 0 and checks["pool_intersect_sealed"] == 0
          and checks["demo_intersect_sealed"] == 0 and checks["quarantine_sealed_subset_of_sealed"]
          and len(train) == 3622)
    out = {"protocol": "sorted(ingest keys) - demo3; random.Random(42).shuffle; first 80",
           "seed": SEED, "demo_keys": list(DEMO_KEYS), "sealed": sealed, "checks": checks,
           "disjoint_from_m9_training": bool(ok)}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    print("disjoint_from_m9_training:", ok, "->", a.out)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
