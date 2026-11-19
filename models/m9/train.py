#!/usr/bin/env python3
"""Re-train the M9_win50 trio (port of the catskills_fec M9 training script with the
``phase3_train_mast_m9.py`` flags ``--seed 42 --epochs 50 --qc --shot-balanced --mask-edge --win``).

Pipeline (all numerics as in the original):

1. Feature cache: for every FAIR-MAST ingest shot, ``X, Y = build_one(shot_from_mast_row(row))``
   -> ``<cache>/<key>.npz`` (X: (T, 308) raw features, Y: (T, 65, 65) EFIT psi, ALL EFIT slices).
2. Split: sorted keys minus the 3 demo shots, ``random.Random(42).shuffle``, first 80 sealed.
3. QC fence on the remaining 3,655: quarantine a shot if max|psi| > median + 10 MAD (also short /
   non-finite / zero-std) -> 3,622 keep.
4. |I_p| window: 2-means (k-means, seed 42) on all keep-shot |I_p| frames; ip_lo = midpoint of the
   centres (328.73 kA on the full pool). Only frames with |I_p| >= ip_lo are trained on.
5. PCA: K = 60, randomized SVD (seed 42, 3 power iterations) on <= 200,000 shot-balanced frames.
6. Train/val: 10 % of the windowed shots (seed-42 shuffle) are validation.
   Features: per-shot temporal augmentation (window 9, smoothdiff) -> 924 columns, standardised
   with train-frame mean/std, clipped at +-10 sigma.
7. Network ``dualctx`` (width 512 x 4 res-blocks, context 512 x 3, dropout 0.05, zero-init gate),
   torch seed = init seed (101 / 202 / 303). Loss: MSE on the 60 PCA coefficients (== pixel MSE of
   the projected target, the basis being orthonormal). AdamW(lr 1e-3, wd 1e-4), grad-clip 1.0,
   batch 2048 drawn shot-balanced (uniform shot, then uniform frame), steps/epoch =
   ceil(n_train_frames / 2048), 50 epochs, keep the best validation epoch. cuDNN disabled.

Note: retraining reproduces the recipe, not the checkpoint bytes (GPU non-determinism, library
versions). ``models/m9/predict.py`` verifies the reference sha256 by default.

    python -m models.m9.train --build-cache                     # step 1 (repo-local cache)
    python -m models.m9.train --member all                      # steps 2-7, 3 members
    python -m models.m9.train --feature-cache /path/to/mast_features_v1 --member 101   # reuse a cache
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.m9.features import IP_COL, build_one, shot_from_mast_row  # noqa: E402
from pipeline.paths import DATA, assert_not_on_lab, fair_mast_dir  # noqa: E402

K = 60
N_PIX = 65 * 65
MAXABS_K = 10.0
MIN_FRAMES = 2
MEMBERS = {101: "M9_win50_i101", 202: "M9_win50_i202", 303: "M9_win50_i303"}
DEFAULT_CACHE = DATA / "fair_mast" / "cache" / "mast_features_v1"


# ------------------------------------------------------------------ 1. feature cache

def _cache_one(args):
    src, dest = args
    from models.io import row_from_parquet

    if Path(dest).is_file():
        return Path(src).stem, "cached"
    X, Y = build_one(shot_from_mast_row(row_from_parquet(Path(src))))
    tmp = Path(dest).with_suffix(".part.npz")
    np.savez(tmp, X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32))
    tmp.replace(dest)
    return Path(src).stem, f"{len(X)} frames"


def build_cache(cache: Path, workers: int) -> None:
    cache = assert_not_on_lab(cache)
    cache.mkdir(parents=True, exist_ok=True)
    srcs = sorted(fair_mast_dir().glob("mast_shot_*.parquet"))
    jobs = [(str(p), str(cache / f"{p.stem}.npz")) for p in srcs]
    import multiprocessing as mp

    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for i, (k, msg) in enumerate(ex.map(_cache_one, jobs, chunksize=4)):
            if (i + 1) % 200 == 0:
                print(f"  [cache] {i + 1}/{len(jobs)} {k} {msg}", flush=True)
    print(f"[cache] {len(jobs)} shots in {cache}")


def load_shot(cache: Path, key: str):
    p = cache / f"{key}.npz"
    if not p.is_file():
        return None
    with np.load(p) as d:
        return np.asarray(d["X"], np.float32), np.asarray(d["Y"], np.float32)


# ------------------------------------------------------------------ 2-4. split, QC, window

def split(cache: Path, seed: int = 42):
    sys.path.insert(0, str(DATA / "fair_mast"))
    from make_holdout import split as _split  # noqa: PLC0415

    keys = sorted(p.stem for p in cache.glob("mast_shot_*.npz"))
    return _split(keys, seed)


def shot_y_stats(cache: Path, key: str) -> dict | None:
    got = load_shot(cache, key)
    if got is None:
        return None
    Y = np.asarray(got[1], np.float64)
    fin = np.isfinite(Y)
    if not fin.any():
        return {"key": key, "n_frames": int(Y.shape[0]), "maxabs": float("nan"), "ystd": float("nan"),
                "finite_frac": 0.0}
    y = Y[fin]
    return {"key": key, "n_frames": int(Y.shape[0]), "ymin": float(y.min()), "ymax": float(y.max()),
            "maxabs": float(np.max(np.abs(y))), "ystd": float(y.std()), "finite_frac": float(fin.mean())}


def qc_keep(cache: Path, pool: list[str]) -> tuple[list[str], dict]:
    stats = [s for k in pool if (s := shot_y_stats(cache, k)) is not None]
    amax = np.asarray([s["maxabs"] for s in stats if np.isfinite(s["maxabs"])], np.float64)
    med = float(np.median(amax))
    mad = max(float(np.median(np.abs(amax - med))), 1e-12)
    thr = {"maxabs_median": med, "maxabs_mad": mad, "maxabs_k": MAXABS_K,
           "maxabs_hi": med + MAXABS_K * mad, "min_frames": MIN_FRAMES, "n_fit": int(amax.size)}

    def reason(s):
        if s["n_frames"] < MIN_FRAMES:
            return "short"
        if not np.isfinite(s["maxabs"]) or s["finite_frac"] < 1.0:
            return "nonfinite"
        if float(s["ystd"]) <= 0.0 or not np.isfinite(s["ystd"]):
            return "zero_std"
        if float(s["maxabs"]) > thr["maxabs_hi"]:
            return "maxabs"
        return None

    return [s["key"] for s in stats if reason(s) is None], thr


def fit_ip_lo(cache: Path, keep: list[str]) -> dict:
    from sklearn.cluster import KMeans

    ip = np.concatenate([np.abs(load_shot(cache, k)[0][:, IP_COL].astype(np.float64)) for k in keep])
    ip = ip[np.isfinite(ip)]
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(ip.reshape(-1, 1))
    c = np.sort(km.cluster_centers_.ravel())
    return {"ip_col": IP_COL, "centers": [float(c[0]), float(c[1])], "ip_lo": float(0.5 * (c[0] + c[1])),
            "n_fit": int(ip.size)}


# ------------------------------------------------------------------ 5-7. PCA + network

def load_matrix(cache: Path, keys: list[str], ip_lo: float):
    Xs, Ys, ids = [], [], []
    for i, key in enumerate(keys):
        got = load_shot(cache, key)
        if got is None:
            continue
        X, Y = got
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        keep = np.abs(np.asarray(X[:, IP_COL], np.float64)) >= float(ip_lo)
        if not keep.any():
            continue
        X, Y = X[keep], Y[keep]
        Xs.append(X)
        Ys.append(np.asarray(Y, np.float32).reshape(len(X), N_PIX))
        ids.append(np.full(len(X), i, np.int32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(ids), Xs[0].shape[1]


def train_member(cache: Path, train_keys: list[str], ip_lo: float, init_seed: int, *, seed: int = 42,
                 epochs: int = 50, batch: int = 2048, device: str = "cuda") -> dict:
    import torch
    from sklearn.decomposition import PCA

    from models.m9.network import _clip_features, augment_temporal, build_dualctx

    torch.backends.cudnn.enabled = False
    dev = torch.device(device)
    t0 = time.perf_counter()
    X, Yflat, ids, n_raw = load_matrix(cache, train_keys, ip_lo)
    rng = np.random.default_rng(seed)
    sample_n = min(200_000, len(Yflat))
    picks = []
    n_shots = int(len(np.unique(ids)))
    for sid in np.unique(ids):
        ix = np.flatnonzero(ids == sid)
        n = max(1, sample_n // n_shots)
        picks.append(rng.choice(ix, size=min(n, len(ix)), replace=False))
    pick = np.concatenate(picks)
    if len(pick) > sample_n:
        pick = rng.choice(pick, size=sample_n, replace=False)
    pca = PCA(n_components=K, svd_solver="randomized", random_state=seed, iterated_power=3)
    pca.fit(Yflat[np.sort(pick)])
    Z = ((Yflat - pca.mean_) @ pca.components_.T).astype(np.float32)
    print(f"[m9] PCA k={K} explained={float(pca.explained_variance_ratio_.sum()):.6f} frames={len(X):,} "
          f"shots={n_shots}", flush=True)

    window, mode, clip = 9, "smoothdiff", 10.0
    shots = np.unique(ids)
    rng.shuffle(shots)
    n_val = max(1, int(0.1 * len(shots)))
    val_m = np.isin(ids, shots[:n_val])
    X_aug = augment_temporal(X, ids, window, mode)
    X64 = np.asarray(X_aug[~val_m], dtype=np.float64)
    mean = np.mean(X64, axis=0)
    std = np.std(X64, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    Xs = _clip_features(((X_aug - mean) / std).astype(np.float32), clip)
    xs = torch.from_numpy(np.ascontiguousarray(Xs))
    zt = torch.from_numpy(Z)

    torch.manual_seed(int(init_seed))
    model = build_dualctx(Xs.shape[1], K, n_raw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_i, va_i = np.flatnonzero(~val_m), np.flatnonzero(val_m)
    tr_by_shot = [a for a in (tr_i[ids[tr_i] == s] for s in np.unique(ids[tr_i])) if len(a)]
    n_steps = max(1, int(math.ceil(len(tr_i) / batch)))
    best = {"val": 1e30, "ep": -1, "state": None}
    hist = []
    for ep in range(epochs):
        model.train()
        for _ in range(n_steps):
            sid = rng.integers(0, len(tr_by_shot), size=batch)
            idx = np.array([int(rng.choice(tr_by_shot[int(s)])) for s in sid], dtype=np.int64)
            pred = model(xs[torch.from_numpy(idx)].to(dev, non_blocking=True))[0]
            loss = ((pred - zt[torch.from_numpy(idx)].to(dev)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pz = torch.cat([model(xs[va_i][s:s + 4096].to(dev))[0].float().cpu()
                            for s in range(0, len(va_i), 4096)]).numpy()
        val = float(((pz - Z[va_i]) ** 2).mean())
        hist.append(val)
        if val < best["val"]:
            best = {"val": val, "ep": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"[m9] seed {init_seed} ep {ep:3d} val mse {val:.6e}{' *' if ep == best['ep'] else ''}", flush=True)
    return {
        "model_state": best["state"], "feature_mean": mean, "feature_std": std,
        "n_raw_features": int(n_raw), "n_features": int(Xs.shape[1]), "n_pca": K,
        "pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32),
        "arch": "dualctx", "temporal_window": window, "temporal_mode": mode, "feature_clip": clip,
        "best_epoch": best["ep"], "val_mse": best["val"], "val_history": hist,
        "train_shots": len(train_keys), "seed": seed, "init_seed": int(init_seed),
        "cache_version": "mast_features_v1", "residual_vac": False, "qc": True, "shot_balanced": True,
        "mask_edge": True, "win": True, "win_ip_lo": float(ip_lo),
        "holdout_excluded": ["mast_shot_28348", "mast_shot_28350", "mast_shot_28351"],
        "epochs": int(epochs), "train_seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--member", default="all", help="101 | 202 | 303 | all")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-shots", type=int, default=0, help="first N train shots only (smoke tests)")
    ap.add_argument("--out", type=Path, default=REPO / "models" / "m9" / "weights")
    a = ap.parse_args()
    if a.build_cache:
        build_cache(a.feature_cache, a.workers)
        return 0
    import torch

    out = assert_not_on_lab(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sealed, pool = split(a.feature_cache)
    keep, thr = qc_keep(a.feature_cache, pool)
    ship_keep = set((DATA / "fair_mast" / "mast_qc_keep.txt").read_text().split())
    train = [k for k in pool if k in set(keep)]
    print(f"[m9] sealed {len(sealed)}  pool {len(pool)}  qc keep {len(keep)} "
          f"(== keep list: {set(keep) == ship_keep})  maxabs_hi {thr['maxabs_hi']:.6g}", flush=True)
    if a.train_shots:
        train = train[:a.train_shots]
    win = fit_ip_lo(a.feature_cache, train)
    print(f"[m9] ip_lo {win['ip_lo']:.10g} kA (fixed 328.7293519015368)", flush=True)
    seeds = list(MEMBERS) if a.member == "all" else [int(a.member)]
    for s in seeds:
        ck = train_member(a.feature_cache, train, win["ip_lo"], s, epochs=a.epochs)
        ck.update({"qc_thresholds": thr, "win_index": win})
        dest = out / f"CRC_FEQ_mast_{MEMBERS[s]}.pt"
        torch.save(ck, dest)
        print(f"[m9] wrote {dest} best epoch {ck['best_epoch']} val {ck['val_mse']:.4e}", flush=True)
        (out / f"{MEMBERS[s]}.json").write_text(json.dumps(
            {k: v for k, v in ck.items() if k in ("best_epoch", "val_mse", "train_shots", "seed", "init_seed",
                                                  "win_ip_lo", "epochs", "train_seconds")}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
