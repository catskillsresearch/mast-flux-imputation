#!/usr/bin/env python3
"""Score MAST predictions with the Fusion Equilibrium Challenge metric (scorer v3.1.0).

The functionals are the organisers' own modules, vendored byte-identical in ``fec_scoring/``
(flat imports, exactly as on Codabench). The per-shot worker and the fold reduction are ported
from catskills_fec ``local_score.py`` / ``catskills/mast_eval.py`` with MACHINE = "MAST":

    S = 0.55 max(0, R2_psi) + 0.15 max(0, (R2_q95 + R2_betaN)/2)
      + 0.10 (1 - min(1, D_LCFS)) + 0.20 mean_j max(0, R2_cons_j)

* every R2 is pooled over the fold, SS_tot about the ground-truth mean;
* one fold-global flux sign s in {+1,-1} minimising SS_res(psi); orientation-sensitive
  extraction runs with axis_sign = AXIS_SIGN[MAST] * s = +s;
* non-finite predicted pixels / scalars are replaced by the GT mean; a failed predicted LCFS
  scores D = 1; frames whose *reference* LCFS fails are dropped from D_LCFS;
* D_LCFS = mean over shots of the per-shot mean of min(1, Hausdorff(LCFS_pred, LCFS_gt) / R_geo).

    python -m pipeline.score outputs/holdout_hybrid.npz
    python -m pipeline.score outputs/holdout_osf.npz --json outputs/holdout_osf.score.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FEC = REPO / "fec_scoring"
if str(FEC) not in sys.path:
    sys.path.insert(0, str(FEC))

from common import (AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS,  # noqa: E402
                    SCALARS)
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

MACHINE = "MAST"
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")
SCORING_VERSION = "3.1.0"


def load_env():
    """MAST grid + vessel mask from the vendored scorer envelope."""
    z = np.load(FEC / "masks" / "mast_envelope.npz")
    R = np.asarray(z["grid_R"], dtype=np.float64).reshape(-1)
    Z = np.asarray(z["grid_Z"], dtype=np.float64).reshape(-1)
    mask = np.asarray(z["mask_coarse"]).astype(bool)
    return R, Z, mask, mask.astype(np.float64)


# ------------------------------------------------------------------ reference (organiser side)

def build_reference(psi_gt, R, Z, mask_coarse, mask_f, machine: str = MACHINE):
    """Per-frame reference LCFS, the seven derived scalars and R_geo (local_score.build_reference)."""
    T = psi_gt.shape[0]
    contours, rgeos = [], []
    cons = np.full((T, N_CONS), np.nan)
    for k in range(T):
        c = extract_lcfs(psi_gt[k], R, Z, machine, mask_coarse, mask_f, n_points=N_POINTS)
        contours.append(c)
        if c is not None:
            rgeos.append(major_radius(c))
        vals = derive_frame(psi_gt[k], R, Z, machine, mask_coarse, mask_f, contour=c)
        for j, name in enumerate(CONS_SCALARS):
            cons[k, j] = vals[name]
    return contours, cons, np.isfinite(cons), (float(np.mean(rgeos)) if rgeos else float("nan"))


# ------------------------------------------------------------------ per-shot worker

def psi_residuals(psi_gt, pred, mean_psi):
    g = psi_gt.astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != g.shape:
        r = g - mean_psi
        ss = float(np.sum(r * r))
        return {s: ss for s in PSI_SIGNS}, 0, True
    nf = ~np.isfinite(p)
    out = {}
    for s in PSI_SIGNS:
        q = np.where(nf, mean_psi, s * p) if nf.any() else s * p
        r = g - q
        out[s] = float(np.sum(r * r))
    return out, int(nf.sum()), False


def score_shot(gt, ref, pred, R, Z, mask_coarse, mask_f, psi_sign, means, machine: str = MACHINE):
    """One shot's contribution (local_score.score_shot, unchanged)."""
    mean_psi, mean_scal, mean_cons = means
    contours, cons_gt, cmask, rgeo = ref
    psi_gt = gt["psi"]
    T = psi_gt.shape[0]
    axis_sign = AXIS_SIGN[machine] * psi_sign

    ss_res_psi, nonfinite, missing = psi_residuals(psi_gt, pred["psirz"], mean_psi)
    ss_tot_shot = float(np.sum((psi_gt.astype(np.float64) - mean_psi) ** 2))
    shot_r2 = {s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0) for s in PSI_SIGNS}

    ss_res_scal = np.zeros(N_SCALARS)
    for j, name in enumerate(SCALARS):
        g = np.asarray(gt[name], dtype=np.float64)
        defined = np.isfinite(g)
        if not defined.any():
            continue
        p = np.asarray(pred.get(name), dtype=np.float64) if pred.get(name) is not None else None
        if p is None or p.shape != (T,):
            ss_res_scal[j] = float(np.sum((g[defined] - mean_scal[j]) ** 2))
        else:
            pj = np.where(np.isfinite(p[defined]), p[defined], mean_scal[j])
            ss_res_scal[j] = float(np.sum((g[defined] - pj) ** 2))

    ss_res_cons = np.zeros(N_CONS)
    cons_frames = np.zeros(N_CONS, dtype=np.int64)
    cons_fails = np.zeros(N_CONS, dtype=np.int64)
    ds, n_fail, n_frames = [], 0, 0
    ref_dropped = sum(1 for c in contours if c is None)

    if missing:
        n_frames = T - ref_dropped
        n_fail = n_frames
        ds = [1.0] * n_frames
        for j in range(N_CONS):
            m = cmask[:, j]
            if m.any():
                ss_res_cons[j] = float(np.sum((cons_gt[m, j] - mean_cons[j]) ** 2))
                cons_frames[j] = cons_fails[j] = int(m.sum())
    else:
        psi_pred = np.asarray(pred["psirz"], dtype=np.float64)
        for k in range(T):
            ref_c, need = contours[k], cmask[k]
            if ref_c is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(psi_pred[k], R, Z, machine, mask_coarse, mask_f,
                                           n_iter=N_ITER, axis_sign=axis_sign)
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(psi_pred[k], R, Z, machine, mask_coarse, mask_f, contour=ex,
                                    with_li=bool(need[LI_IDX]), axis_sign=axis_sign)
                for j in np.nonzero(need)[0]:
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v = mean_cons[j]
                        cons_fails[j] += 1
                    dv = cons_gt[k, j] - v
                    ss_res_cons[j] += dv * dv

    return {
        "ss_res_psi": ss_res_psi, "psi_pixels": int(psi_gt.size), "psi_nonfinite": nonfinite,
        "ss_res_scal": ss_res_scal, "ss_res_cons": ss_res_cons,
        "cons_frames": cons_frames, "cons_fails": cons_fails,
        "shot_r2_psi": shot_r2, "shot_dlcfs": (float(np.mean(ds)) if ds else None),
        "lcfs_frames": n_frames, "lcfs_fails": n_fail, "ref_lcfs_dropped": ref_dropped,
        "missing": missing,
    }


# ------------------------------------------------------------------ process pool plumbing

def _init_worker() -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _ref_task(args):
    psi, R, Z, mc, mf = args
    return build_reference(psi, R, Z, mc, mf)


def _score_task(args):
    return score_shot(*args)


def _pmap(fn, payloads, workers):
    if workers <= 1:
        return [fn(p) for p in payloads]
    import multiprocessing as mp

    with ProcessPoolExecutor(min(workers, len(payloads)), mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as ex:
        return list(ex.map(fn, payloads, chunksize=1))


# ------------------------------------------------------------------ fold scoring

def score_fold(gts: list[dict], preds: list[dict], workers: int = 8) -> tuple[dict, list[dict]]:
    """Official MAST composite over a fold. Returns (result, per-shot parts)."""
    R, Z, mask_coarse, mask_f = load_env()
    refs = _pmap(_ref_task, [(g["psi"], R, Z, mask_coarse, mask_f) for g in gts], workers)

    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = (np.zeros(N_SCALARS) for _ in range(3))
    cons_sum, cons_sumsq, cons_n = (np.zeros(N_CONS) for _ in range(3))
    for gt, ref in zip(gts, refs):
        g = gt["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum())
        psi_sumsq += float((g[fin] ** 2).sum())
        psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(gt[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum()
            scal_sumsq[j] += (v ** 2).sum()
            scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum()
            cons_sumsq[j] += (v ** 2).sum()
            cons_n[j] += v.size
    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    totals = {s: 0.0 for s in PSI_SIGNS}
    for gt, p in zip(gts, preds):
        rr, _, _ = psi_residuals(gt["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    parts = _pmap(_score_task, [(gt, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means, MACHINE)
                                for gt, ref, p in zip(gts, refs, preds)], workers)
    acc = Accum()
    acc.psi_sign = psi_sign
    for part in parts:
        acc.add(part)
    res = finalize_machine(acc, ref_stats)
    res["n_frames"] = int(sum(g["psi"].shape[0] for g in gts))
    res["scoring_version"] = SCORING_VERSION
    return res, parts


def load_predictions(npz: Path, n: int) -> list[dict]:
    z = np.load(npz, allow_pickle=False)
    return [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64),
             "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
             "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)} for i in range(n)]


def components(res: dict) -> dict:
    return {"S": res["S"], "R2_psi": res["r2_psi"], "R2_q95": res["r2_each"]["q95"],
            "R2_betaN": res["r2_each"]["betaN"], "R2_qb": res["r2_qb"],
            "one_minus_D_LCFS": round(1.0 - min(1.0, res["dlcfs"]), 6),
            "Consistency": res["consistency"], **{f"R2_{k}": v for k, v in res["r2_cons_each"].items()}}


def print_table(res: dict, title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {title}")
    print(f"  COMPOSITE S = {res['S']:.6f}   ({res['n_shots']} shots, {res['n_frames']} frames)")
    print("=" * 66)
    for label, v, w in (("R2_psi", res["r2_psi"], 0.55), ("R2_{q95,betaN}", res["r2_qb"], 0.15),
                        ("1 - D_LCFS", 1.0 - min(1.0, res["dlcfs"]), 0.10),
                        ("Consistency", res["consistency"], 0.20)):
        print(f"  {label:>16s}  {v:9.6f}   x {w:.2f}  =  {w * max(0.0, v):.6f}")
    print(f"      R2_q95 {res['r2_each']['q95']:9.6f}   R2_betaN {res['r2_each']['betaN']:9.6f}")
    print("  per-derived-scalar R2 (Consistency):")
    for k, v in res["r2_cons_each"].items():
        print(f"      {k:>8s}  {'  n/a' if v is None else f'{v:9.6f}'}")
    print(f"  LCFS fail {res['lcfs_fail_frac']:.4%}  derive fail {res['cons_fail_frac']:.4%}  "
          f"psi_sign {res['psi_sign']:+d}  ref LCFS dropped {res['ref_lcfs_dropped']}")


def main() -> int:
    from pipeline.datasets import load_shot
    from pipeline.paths import assert_not_on_lab

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path, help="npz written by pipeline.predict (with its .json sidecar)")
    ap.add_argument("--json", type=Path, default=None, help="default <pred>.score.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    meta = json.loads(a.pred.with_suffix(".json").read_text())
    split, keys = meta["split"], meta["keys"]
    if split == "public_test":
        raise SystemExit("public_test has no labels; nothing to score")
    t0 = time.time()
    shots = [load_shot(split, k) for k in keys]
    gts = [s.gt for s in shots]
    preds = load_predictions(a.pred, len(keys))
    for k, g, p in zip(keys, gts, preds):
        if p["psirz"].shape != g["psi"].shape:
            raise SystemExit(f"{k}: prediction {p['psirz'].shape} vs GT {g['psi'].shape}")
    res, parts = score_fold(gts, preds, a.workers)
    title = f"{split} / {meta['mode']}  ({a.pred.name})"
    print_table(res, title)
    per_shot = [{"key": k, "T": int(g["psi"].shape[0]),
                 "r2_psi": round(float(pt["shot_r2_psi"][res["psi_sign"]]), 6),
                 "dlcfs": (None if pt["shot_dlcfs"] is None else round(pt["shot_dlcfs"], 6)),
                 "lcfs_fails": int(pt["lcfs_fails"]), "lcfs_frames": int(pt["lcfs_frames"])}
                for k, g, pt in zip(keys, gts, parts)]
    out = {"split": split, "mode": meta["mode"], "pred": str(a.pred), "components": components(res),
           "result": res, "excluded_no_diverted_frames": meta.get("excluded_no_diverted_frames", []),
           "per_shot": per_shot, "seconds": round(time.time() - t0, 1)}
    js = assert_not_on_lab(a.json or a.pred.with_suffix(".score.json"))
    js.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n[score] wrote {js} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
