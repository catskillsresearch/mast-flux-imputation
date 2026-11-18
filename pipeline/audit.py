#!/usr/bin/env python3
"""Failure-mode audit: the shot viewer's per-frame / per-shot diagnostics as automated metrics.

For one prediction file (``outputs/<split>_<mode>.npz`` + ``.json`` sidecar, as written by
``pipeline.predict``) and its ground truth (``pipeline.datasets.load_shot``), every frame gets

* psi:      frame R^2 (vessel mask), frame MSE / RMSE, Ampere I_p ratio (pred / EFIT), and the share
            of the residual outside a PCA subspace (``--pca`` checkpoint, optional);
* LCFS:     symmetric Hausdorff distance [m] and D = min(1, H / R_geo) (the FEC term), area of each
            boundary and the area of their symmetric difference / EFIT area, LCFS-extraction failure;
* points:   O-point displacement, X-point displacement per half-plane, X-point lost / gained,
            diverted <-> limited flip (an X-point within 5 cm of the LCFS = diverted);
* jump:     the viewer's flag: max(|dZ_top|, |dZ_bottom|) of the LCFS tips >= 0.8 grid cells (50 mm),
            or a missing predicted LCFS where EFIT has one;
* scalars:  the seven FEC derived scalars (R/Z axis, kappa, delta_top/bot, V, l_i) of prediction and
            EFIT, plus q95 and beta_N against the EFIT values;
* tags:     failure classes (see ``TAGS``).

Per shot it adds temporal checks on every tracked series (lag by cross-correlation within +-4
frames, jitter = rms of second differences pred / EFIT, drift = end-to-start change of the
residual from a linear fit), and over the run the official FEC composite S (``pipeline.score``).
All geometry uses the vendored scorer modules and the viewer's own helpers.

    python -m pipeline.audit outputs/demo_hybrid.npz --tag demo_hybrid_shipped [--pca auto]
    python -m pipeline.audit compare outputs/audit/OLD outputs/audit/NEW   # new >= old check

Writes ``outputs/audit/<tag>/{frames.csv, shots.csv, summary.json, summary.md}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "fec_scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import CONS_SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame, magnetic_axis  # noqa: E402
from lcfs import extract_lcfs_with_sign, major_radius  # noqa: E402

MACHINE = "MAST"
MU0 = 4.0e-7 * np.pi
JUMP_CELLS = 0.8
XPT_NEAR_LCFS = 0.05      # m: X-point this close to the LCFS -> diverted
AXIS_SHIFT = 0.05         # m
XPT_SHIFT = 0.05          # m
AREA_TOL = 0.05           # relative
KAPPA_TOL = 0.10
SERIES = ["q95", "betaN", *CONS_SCALARS, "lcfs_zmax", "lcfs_zmin"]
TAGS = ("lcfs_fail", "lcfs_wide", "lcfs_narrow", "axis_shift", "xpt_lost", "xpt_gained", "xpt_shift",
        "topology_flip", "jump", "kappa_bias")


def _env():
    z = np.load(REPO / "fec_scoring" / "masks" / "mast_envelope.npz")
    R = np.asarray(z["grid_R"], float).reshape(-1)
    Z = np.asarray(z["grid_Z"], float).reshape(-1)
    m = np.asarray(z["mask_coarse"]).astype(bool)
    return R, Z, m, m.astype(float)


R, Z, MASK, MASK_F = _env()
_FR, _FZ = np.meshgrid(np.arange(0.0, 2.0, 0.01) + 0.005, np.arange(-2.0, 2.0, 0.01) + 0.005)
FINE = np.column_stack([_FR.ravel(), _FZ.ravel()])


# ------------------------------------------------------------------ geometry (viewer helpers)

def lcfs(psi):
    c, _ = extract_lcfs_with_sign(psi, R, Z, MACHINE, MASK, MASK_F, n_points=256, axis_sign=+1.0)
    return None if c is None else np.asarray(c, float)


def opoint(psi):
    ra, za, _, _ = magnetic_axis(psi, R, Z, MACHINE, MASK, +1.0)
    return float(ra), float(za)


def xpoints(psi):
    """{'upper': (R, Z) | None, 'lower': ...}; same saddle search as app/fec_site."""
    dz, dr = Z[1] - Z[0], R[1] - R[0]
    gz, gr = np.gradient(psi, dz, dr)
    g2 = gz * gz + gr * gr
    gzz, gzr = np.gradient(gz, dz, dr)
    _, grr = np.gradient(gr, dz, dr)
    det = gzz * grr - gzr * gzr
    out = {}
    for half, upper in (("upper", True), ("lower", False)):
        best, bv = None, np.inf
        for iz in range(2, len(Z) - 2):
            if (Z[iz] > 0.3) != upper or abs(Z[iz]) <= 0.3:
                continue
            for ir in range(2, len(R) - 2):
                if not MASK[iz, ir] or det[iz, ir] >= 0:
                    continue
                v = g2[iz, ir]
                if v <= g2[iz - 1:iz + 2, ir - 1:ir + 2].min() and v < bv:
                    best, bv = (float(R[ir]), float(Z[iz])), v
        out[half] = best
    return out


def ampere_ip(psi):
    dpz, dpr = np.gradient(psi, Z, R)
    rg = np.maximum(R[None, :], 1e-6)
    _, d_inv = np.gradient(dpr / rg, Z, R)
    d2z, _ = np.gradient(dpz, Z, R)
    j = (rg * d_inv + d2z) / (MU0 * rg)
    return float(np.nansum(j) * np.mean(np.diff(R)) * np.mean(np.diff(Z)))


def poly_area(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def inside(c):
    from matplotlib.path import Path as MPath
    return MPath(c).contains_points(FINE)


def dist_to_poly(pt, c):
    if pt is None or c is None:
        return np.inf
    return float(np.min(np.hypot(c[:, 0] - pt[0], c[:, 1] - pt[1])))


def diverted(xp, c):
    return any(dist_to_poly(xp[h], c) < XPT_NEAR_LCFS for h in ("upper", "lower"))


def _r2(g, p):
    g, p = np.asarray(g, float).ravel(), np.asarray(p, float).ravel()
    m = np.isfinite(g) & np.isfinite(p)
    ss = float(np.sum((g[m] - g[m].mean()) ** 2)) if m.sum() > 1 else 0.0
    return float("nan") if ss <= 0 else 1.0 - float(np.sum((g[m] - p[m]) ** 2)) / ss


# ------------------------------------------------------------------ per shot

def audit_shot(args):
    key, t, gt, pr, pca = args
    T = gt["psi"].shape[0]
    rows = []
    c_gt_all = [lcfs(gt["psi"][k]) for k in range(T)]
    rg = [major_radius(c) for c in c_gt_all if c is not None]
    rgeo = float(np.mean(rg)) if rg else float("nan")
    dz = float(abs(np.mean(np.diff(Z))))
    for k in range(T):
        g, p = gt["psi"][k], pr["psi"][k]
        d = (p - g)[MASK]
        row = {"key": key, "k": k, "t_ms": float(t[k]), "r2_psi_frame": _r2(g[MASK], p[MASK]),
               "mse_psi": float(np.mean(d ** 2)), "rmse_psi": float(np.sqrt(np.mean(d ** 2)))}
        ag = ampere_ip(g)
        row["ampere_ratio"] = ampere_ip(p) / ag if abs(ag) > 1e-30 else float("nan")
        if pca is not None:
            mu, C = pca
            r = (p - g).reshape(-1)
            proj = C.T @ (C @ r)
            row["pca_resid_share"] = float(np.sum((r - proj) ** 2) / max(np.sum(r ** 2), 1e-30))
        cg, cp = c_gt_all[k], lcfs(p)
        row["lcfs_fail"] = int(cg is not None and cp is None)
        if cg is not None and cp is not None:
            h = symmetric_hausdorff(cp, cg)
            row["hausdorff_m"] = float(h)
            row["D_lcfs"] = min(1.0, h / rgeo)
            ig, ip_ = inside(cg), inside(cp)
            ag_ = float(ig.sum()) * 1e-4
            row["area_gt_m2"], row["area_pred_m2"] = poly_area(cg), poly_area(cp)
            row["area_ratio"] = row["area_pred_m2"] / max(row["area_gt_m2"], 1e-9)
            row["area_symdiff_rel"] = float(np.logical_xor(ig, ip_).sum()) * 1e-4 / max(ag_, 1e-9)
            gap = max(abs(cp[:, 1].max() - cg[:, 1].max()), abs(cp[:, 1].min() - cg[:, 1].min()))
            row["tip_gap_m"] = float(gap)
            row["jump"] = int(gap >= JUMP_CELLS * dz)
        else:
            row["D_lcfs"] = 1.0 if cg is not None else float("nan")
            row["jump"] = int(cg is not None and cp is None)
        row["lcfs_zmax_gt"] = float(cg[:, 1].max()) if cg is not None else float("nan")
        row["lcfs_zmin_gt"] = float(cg[:, 1].min()) if cg is not None else float("nan")
        row["lcfs_zmax_pred"] = float(cp[:, 1].max()) if cp is not None else float("nan")
        row["lcfs_zmin_pred"] = float(cp[:, 1].min()) if cp is not None else float("nan")
        og, op = opoint(g), opoint(p)
        row["opoint_disp_m"] = float(math.hypot(op[0] - og[0], op[1] - og[1]))
        xg, xp = xpoints(g), xpoints(p)
        lost = gained = 0
        disp = []
        for hlf in ("upper", "lower"):
            if xg[hlf] is not None and xp[hlf] is None:
                lost += 1
            elif xg[hlf] is None and xp[hlf] is not None:
                gained += 1
            elif xg[hlf] is not None:
                disp.append(math.hypot(xp[hlf][0] - xg[hlf][0], xp[hlf][1] - xg[hlf][1]))
        row["xpt_lost"], row["xpt_gained"] = lost, gained
        row["xpt_disp_m"] = float(np.mean(disp)) if disp else float("nan")
        row["diverted_gt"], row["diverted_pred"] = int(diverted(xg, cg)), int(diverted(xp, cp))
        row["topology_flip"] = int(row["diverted_gt"] != row["diverted_pred"])
        vg = derive_frame(g, R, Z, MACHINE, MASK, MASK_F, contour=cg)
        vp = derive_frame(p, R, Z, MACHINE, MASK, MASK_F, contour=cp if cp is not None else None)
        for s in CONS_SCALARS:
            row[f"{s}_gt"], row[f"{s}_pred"] = float(vg[s]), float(vp[s])
        for s in ("q95", "betaN"):
            row[f"{s}_gt"], row[f"{s}_pred"] = float(gt[s][k]), float(pr[s][k])
        tags = []
        if row["lcfs_fail"]:
            tags.append("lcfs_fail")
        ar = row.get("area_ratio", float("nan"))
        if ar > 1 + AREA_TOL:
            tags.append("lcfs_wide")
        if ar < 1 - AREA_TOL:
            tags.append("lcfs_narrow")
        if row["opoint_disp_m"] > AXIS_SHIFT:
            tags.append("axis_shift")
        if lost:
            tags.append("xpt_lost")
        if gained:
            tags.append("xpt_gained")
        if row["xpt_disp_m"] > XPT_SHIFT:
            tags.append("xpt_shift")
        if row["topology_flip"]:
            tags.append("topology_flip")
        if row["jump"]:
            tags.append("jump")
        if abs(row["kappa_pred"] - row["kappa_gt"]) > KAPPA_TOL:
            tags.append("kappa_bias")
        row["tags"] = ";".join(tags)
        rows.append(row)
    # temporal
    shot = {"key": key, "T": T, "r2_psi_shot": _r2(gt["psi"][:, MASK], pr["psi"][:, MASK]),
            "mean_D_lcfs": float(np.nanmean([r["D_lcfs"] for r in rows]))}
    for s in SERIES:
        a = np.array([r[f"{s}_pred"] for r in rows], float)
        b = np.array([r[f"{s}_gt"] for r in rows], float)
        shot.update(temporal(s, a, b))
    for tg in TAGS:
        shot[f"rate_{tg}"] = float(np.mean([tg in r["tags"].split(";") for r in rows]))
    return rows, shot


def temporal(name, a, b, maxlag=4):
    m = np.isfinite(a) & np.isfinite(b)
    out = {f"{name}_mae": float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")}
    if m.sum() < 8:
        return {**out, f"{name}_lag": float("nan"), f"{name}_jitter": float("nan"), f"{name}_drift": float("nan")}
    a2, b2 = np.interp(np.arange(len(a)), np.flatnonzero(m), a[m]), np.interp(np.arange(len(b)), np.flatnonzero(m), b[m])
    ad, bd = a2 - a2.mean(), b2 - b2.mean()
    best, bl = -np.inf, 0
    for L in range(-maxlag, maxlag + 1):
        x, y = (ad[L:], bd[:len(bd) - L]) if L >= 0 else (ad[:L], bd[-L:])
        den = np.sqrt(np.sum(x * x) * np.sum(y * y))
        c = float(np.sum(x * y) / den) if den > 0 else -np.inf
        if c > best:
            best, bl = c, L
    jg = np.sqrt(np.mean(np.diff(b2, 2) ** 2))
    jp = np.sqrt(np.mean(np.diff(a2, 2) ** 2))
    res = a2 - b2
    slope = np.polyfit(np.arange(len(res)), res, 1)[0]
    return {**out, f"{name}_lag": float(bl), f"{name}_jitter": float(jp / jg) if jg > 0 else float("nan"),
            f"{name}_drift": float(slope * (len(res) - 1))}


# ------------------------------------------------------------------ run

def load_pca(spec):
    if not spec:
        return None
    import torch
    if spec == "auto":
        from pipeline.paths import m9_weights_dir
        spec = str(m9_weights_dir() / "CRC_FEQ_mast_M9_win50_i101.pt")
    ck = torch.load(spec, map_location="cpu", weights_only=False)
    C = np.asarray(ck["pca_components"], np.float64)
    return np.asarray(ck["pca_mean"], np.float64), C


SUMMARY_SPEC = [  # (name, direction) higher=+1, lower=-1
    ("S", +1), ("R2_psi", +1), ("R2_q95", +1), ("R2_betaN", +1), ("one_minus_D_LCFS", +1),
    ("Consistency", +1), *[(f"R2_{s}", +1) for s in CONS_SCALARS],
    ("mean_r2_psi_frame", +1), ("mean_mse_psi", -1), ("mean_hausdorff_mm", -1), ("p95_hausdorff_mm", -1),
    ("mean_area_symdiff_rel", -1), ("mean_opoint_disp_mm", -1), ("mean_xpt_disp_mm", -1),
    ("rate_lcfs_fail", -1), ("rate_xpt_lost", -1), ("rate_topology_flip", -1), ("rate_jump", -1),
    ("mae_q95", -1), ("mae_betaN", -1), ("mae_kappa", -1), ("mae_li", -1),
    ("mean_abs_lag", -1), ("mean_abs_log_jitter", -1),
]


def summarise(frames, shots, fec):
    f = lambda k: np.array([r.get(k, np.nan) for r in frames], float)  # noqa: E731
    out = dict(fec)
    out["n_shots"], out["n_frames"] = len(shots), len(frames)
    out["mean_r2_psi_frame"] = float(np.nanmean(f("r2_psi_frame")))
    out["mean_mse_psi"] = float(np.nanmean(f("mse_psi")))
    h = f("hausdorff_m") * 1e3
    out["mean_hausdorff_mm"] = float(np.nanmean(h))
    out["p95_hausdorff_mm"] = float(np.nanpercentile(h, 95)) if np.isfinite(h).any() else float("nan")
    out["mean_area_symdiff_rel"] = float(np.nanmean(f("area_symdiff_rel")))
    out["mean_opoint_disp_mm"] = float(np.nanmean(f("opoint_disp_m")) * 1e3)
    out["mean_xpt_disp_mm"] = float(np.nanmean(f("xpt_disp_m")) * 1e3)
    out["rate_lcfs_fail"] = float(np.nanmean(f("lcfs_fail")))
    out["rate_xpt_lost"] = float(np.nanmean(f("xpt_lost") > 0))
    out["rate_topology_flip"] = float(np.nanmean(f("topology_flip")))
    out["rate_jump"] = float(np.nanmean(f("jump")))
    for s in ("q95", "betaN", "kappa", "li"):
        out[f"mae_{s}"] = float(np.nanmean(np.abs(f(f"{s}_pred") - f(f"{s}_gt"))))
    lags = [abs(s[f"{n}_lag"]) for s in shots for n in SERIES if np.isfinite(s.get(f"{n}_lag", np.nan))]
    jit = [abs(math.log(s[f"{n}_jitter"])) for s in shots for n in SERIES
           if np.isfinite(s.get(f"{n}_jitter", np.nan)) and s[f"{n}_jitter"] > 0]
    out["mean_abs_lag"] = float(np.mean(lags)) if lags else float("nan")
    out["mean_abs_log_jitter"] = float(np.mean(jit)) if jit else float("nan")
    tags = {}
    for r in frames:
        for tg in filter(None, r["tags"].split(";")):
            tags[tg] = tags.get(tg, 0) + 1
    out["tag_counts"] = dict(sorted(tags.items(), key=lambda kv: -kv[1]))
    return out


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return "n/a" if not np.isfinite(v) else f"{v:.6g}"


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(pred: Path, tag: str, pca_spec: str | None, workers: int) -> Path:
    from pipeline.datasets import load_shot
    from pipeline.paths import assert_not_on_lab, output_dir
    from pipeline.score import components, load_predictions, score_fold
    meta = json.loads(pred.with_suffix(".json").read_text())
    split, keys = meta["split"], meta["keys"]
    shots = [load_shot(split, k) for k in keys]
    preds = load_predictions(pred, len(keys))
    pca = load_pca(pca_spec)
    t0 = time.time()
    tasks = [(k, np.asarray(s.row["efit_times"], float).reshape(-1), s.gt,
              {"psi": p["psirz"], "q95": p["q95"], "betaN": p["betaN"]}, pca)
             for k, s, p in zip(keys, shots, preds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(audit_shot, tasks))
    frames = [r for rows, _ in res for r in rows]
    shot_rows = [s for _, s in res]
    fec, _ = score_fold([s.gt for s in shots], preds, workers)
    summ = summarise(frames, shot_rows, components(fec))
    summ.update({"pred": str(pred), "split": split, "mode": meta.get("mode"), "tag": tag,
                 "seconds": round(time.time() - t0, 1)})
    out = assert_not_on_lab(output_dir() / "audit" / tag)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "frames.csv", frames)
    write_csv(out / "shots.csv", shot_rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    lines = [f"# Audit `{tag}` ({split}/{meta.get('mode')}, {summ['n_shots']} shots, {summ['n_frames']} frames)", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(summ.get(k))} |" for k, _ in SUMMARY_SPEC]
    lines += ["", "Failure tags (frames): " + ", ".join(f"{k} {v}" for k, v in summ["tag_counts"].items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[audit] wrote {out} ({summ['seconds']} s)")
    return out


def compare(old: Path, new: Path) -> int:
    """Side-by-side table; every metric must satisfy new >= old (in its 'better' direction)."""
    a = json.loads((old / "summary.json").read_text())
    b = json.loads((new / "summary.json").read_text())
    print(f"| metric | better | old ({a['tag']}) | new ({b['tag']}) | new >= old |")
    print("|---|---|---|---|---|")
    fails = 0
    for k, d in SUMMARY_SPEC:
        x, y = a.get(k), b.get(k)
        ok = None
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            ok = (y >= x - 1e-12) if d > 0 else (y <= x + 1e-12)
            fails += int(not ok)
        print(f"| {k} | {'higher' if d > 0 else 'lower'} | {_fmt(x)} | {_fmt(y)} | "
              f"{'n/a' if ok is None else ('yes' if ok else 'NO')} |")
    print(f"\n{fails} metric(s) where new < old")
    return int(fails > 0)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        return compare(Path(sys.argv[2]), Path(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pca", default=None, help="'auto' (shipped M9 member i101) or a checkpoint path")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    run(a.pred, a.tag or a.pred.stem, a.pca, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
