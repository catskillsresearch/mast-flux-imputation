#!/usr/bin/env python3
"""Shared helpers for scripts/make_model_report.py: constants, audit readers, curve distances,
the per-shot curve table and figure saving.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E6 = REPO / "outputs" / "experiments" / "E6"
AUD = REPO / "outputs" / "audit"
GEN = REPO / "paper" / "generated"
FIG = REPO / "paper" / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": 150})
NEWC = "#2f8f3a"
TRACES = (("q95", "$q_{95}$"), ("betaN", r"$\beta_N$"), ("kappa", r"$\kappa$"), ("tri_top", r"$\delta_{\rm top}$"),
          ("tri_bot", r"$\delta_{\rm bot}$"), ("li", r"$\ell_i$"), ("volume", r"$V$ (m$^3$)"),
          ("R_axis", r"$R_{\rm axis}$ (m)"), ("Z_axis", r"$Z_{\rm axis}$ (m)"))
METRICS = (("hausdorff_m", "LCFS Hausdorff (mm)", 1000.0, 0.0), ("xpt_disp_m", "X-point offset (mm)", 1000.0, 0.0),
           ("r2_psi_frame", r"$R^2_\psi$ (frame)", 1.0, 1.0))
ROWS = (("S", "$S$"), ("R2_psi", r"$R^2_\psi$"), ("R2_q95", "$R^2_{q_{95}}$"), ("R2_betaN", r"$R^2_{\beta_N}$"),
        ("one_minus_D_LCFS", r"$1-D_{\rm LCFS}$"), ("Consistency", "Consistency"))
CONS = (("R2_R_axis", r"$R_{\rm axis}$"), ("R2_Z_axis", r"$Z_{\rm axis}$"), ("R2_kappa", r"$\kappa$"),
        ("R2_tri_top", r"$\delta_{\rm top}$"), ("R2_tri_bot", r"$\delta_{\rm bot}$"), ("R2_volume", "$V$"),
        ("R2_li", r"$\ell_i$"))
PLAIN = {"S": "S", "R2_psi": "R2_psi", "R2_q95": "R2_q95", "R2_betaN": "R2_betaN", "one_minus_D_LCFS": "1-D_LCFS",
         "Consistency": "Consistency"}


def rj(p):
    return json.loads(Path(p).read_text())


def f6(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):.6f}"


def f4(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):.4f}"


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def frames(tag):
    out = {}
    with open(AUD / tag / "frames.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["key"], []).append(r)
    return {k: sorted(v, key=lambda r: int(r["k"])) for k, v in out.items()}


def curve_distance(rows):
    """Per series: RMSE over frames and time-averaged |model - truth| (area / duration); for the
    nine traces also both divided by the truth's range over the shot (nRMSE, narea)."""
    t_all = [num(r["t_ms"]) for r in rows]
    out = {}
    for q, _ in TRACES:
        pts = [(t, num(r[f"{q}_pred"]), num(r[f"{q}_gt"])) for t, r in zip(t_all, rows)]
        out[q] = _dist(pts, True)
    for q, _, f, ref in METRICS:
        pts = [(t, None if num(r[q]) is None else num(r[q]) * f, ref * f if q != "r2_psi_frame" else ref)
               for t, r in zip(t_all, rows)]
        out[q] = _dist(pts, False)
    return out


def _dist(pts, norm):
    pts = [p for p in pts if None not in p]
    if len(pts) < 2:
        return None
    t = [p[0] for p in pts]
    d = [p[1] - p[2] for p in pts]
    rmse = math.sqrt(sum(x * x for x in d) / len(d))
    area = sum(0.5 * (abs(d[i]) + abs(d[i + 1])) * (t[i + 1] - t[i]) for i in range(len(t) - 1)) / max(t[-1] - t[0], 1e-9)
    rec = {"rmse": rmse, "area": area}
    if norm:
        g = [p[2] for p in pts]
        rng = max(g) - min(g)
        if rng > 0:
            rec["nrmse"], rec["narea"] = rmse / rng, area / rng
    return rec


LABELS = {**{q: l for q, l in TRACES}, **{q: l for q, l, _, _ in METRICS}}




def t_curves(fr):
    keys = sorted(fr)
    L = ["| curve | " + " | ".join(f"{k.replace('mast_shot_', '')} RMSE | {k.replace('mast_shot_', '')} area" for k in keys) + " |",
         "|---|" + "---|---|" * len(keys)]
    cds = {k: curve_distance(fr[k]) for k in keys}
    for q in [q for q, _ in TRACES] + [q for q, _, _, _ in METRICS]:
        cells = []
        for k in keys:
            r = cds[k].get(q)
            if r is None:
                cells += ["–", "–"]
            elif "nrmse" in r:
                cells += [f"{r['rmse']:.4g} ({100 * r['nrmse']:.1f}%)", f"{r['area']:.4g} ({100 * r['narea']:.1f}%)"]
            else:
                cells += [f"{r['rmse']:.4g}", f"{r['area']:.4g}"]
        L.append(f"| {LABELS[q]} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n", cds




def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)

