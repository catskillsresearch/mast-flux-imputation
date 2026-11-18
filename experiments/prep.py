"""Common training data for all arms: cleaned pool, fixed |Ip| window, PCA basis, splits.

* pool = shipped sealed-80 split minus the 349 no-plasma shots (outputs/compat/no_plasma.json);
* QC recomputed on that pool (shipped rule); |Ip| window fixed at the shipped 328.7293519015368 kA;
* PCA K=60 exactly as shipped (shot-balanced <= 200k frames, randomized SVD seed 42);
* archive validation = first 10 % of the contributing shots after random.Random(42).shuffle;
  archval100 / select30 = its first 100 / 30 shots; osftrain200 = 200 training shots (seed 43).
Writes outputs/experiments/common/data.npz and data/fair_mast/experiment_splits_seed42.json.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np

from experiments.common import COMMON, LAB_CACHE, REPO, SHIPPED_IP_LO, SPLITS_JSON, assert_safe
from models.m9 import train as T
from models.m9.features import IP_COL


def main() -> dict:
    t0 = time.time()
    out = assert_safe(COMMON)
    out.mkdir(parents=True, exist_ok=True)
    sealed, pool = T.split(LAB_CACHE)
    nop = set(json.loads((REPO / "outputs/compat/no_plasma.json").read_text())["no_plasma_keys"])
    pool2 = [k for k in pool if k not in nop]
    keep, thr = T.qc_keep(LAB_CACHE, pool2)
    ship = set((REPO / "data/fair_mast/mast_qc_keep.txt").read_text().split())
    Xs, Ys, ids, fis, keys = [], [], [], [], []
    for key in keep:
        got = T.load_shot(LAB_CACHE, key)
        if got is None:
            continue
        X, Y = got
        n = min(len(X), len(Y))
        m = np.abs(X[:n, IP_COL].astype(np.float64)) >= SHIPPED_IP_LO
        if not m.any():
            continue
        Xs.append(X[:n][m])
        Ys.append(Y[:n][m].reshape(int(m.sum()), -1).astype(np.float32))
        ids.append(np.full(int(m.sum()), len(keys), np.int32))
        fis.append(np.flatnonzero(m).astype(np.int32))
        keys.append(key)
    X = np.concatenate(Xs); del Xs
    Y = np.concatenate(Ys); del Ys
    ids = np.concatenate(ids); fi = np.concatenate(fis)
    rng = np.random.default_rng(42)
    n_shots = len(keys)
    per = max(1, 200_000 // n_shots)
    picks = []
    for s in range(n_shots):
        ix = np.flatnonzero(ids == s)
        picks.append(rng.choice(ix, size=min(per, len(ix)), replace=False))
    pick = np.concatenate(picks)
    if len(pick) > 200_000:
        pick = rng.choice(pick, size=200_000, replace=False)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=T.K, svd_solver="randomized", random_state=42, iterated_power=3)
    pca.fit(Y[np.sort(pick)])
    Z = ((Y - pca.mean_) @ pca.components_.T).astype(np.float32)
    del Y
    order = list(keys)
    random.Random(42).shuffle(order)
    n_val = max(1, int(0.1 * len(order)))
    archval = order[:n_val]
    train_keys = sorted(set(keys) - set(archval))
    osftrain = sorted(random.Random(43).sample(train_keys, 200))
    SPLITS_JSON.write_text(json.dumps({"archval": archval, "archval100": archval[:100], "select30": archval[:30],
                                       "osftrain200": osftrain}, indent=0) + "\n")
    np.savez(out / "data.npz", X=X, Z=Z, ids=ids, fi=fi, keys=np.array(keys),
             pca_mean=pca.mean_.astype(np.float32), pca_components=pca.components_.astype(np.float32),
             ip_lo=np.float64(SHIPPED_IP_LO), archval=np.array(archval))
    info = {"pool": len(pool), "no_plasma_removed": len(pool) - len(pool2), "qc_keep": len(keep),
            "qc_keep_minus_shipped": len(set(keep) - ship), "shipped_minus_qc_keep": len(ship - set(keep) - nop),
            "contributing_shots": len(keys), "frames": int(len(X)), "archval_shots": len(archval),
            "explained": float(pca.explained_variance_ratio_.sum()), "qc_thr": thr, "seconds": round(time.time() - t0, 1)}
    (out / "prep.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    print("[prep]", info, flush=True)
    return info


if __name__ == "__main__":
    main()
