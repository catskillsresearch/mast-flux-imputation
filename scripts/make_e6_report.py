#!/usr/bin/env python3
"""E6 tables and figures for the paper, from saved E6 outputs only (no numbers typed in).

Inputs: outputs/experiments/E6/{final,select,s1,sweep_choice}.json, m*/summary.json, the audits
outputs/audit/exp_E6_{demo,holdout}_hybrid, exp_E4_*, and the E6 demo predictions.
Paper outputs (ground truth vs the E6 calibration): paper/generated/e6_*.md, paper/figures/e6_*.{pdf,png}.
Internal check (not in the paper): outputs/experiments/E6/internal_check.md compares E6 with the
FEC Phase-2 submission (audit tag <split>_hybrid_shipped) row by row and curve by curve.

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/make_e6_report.py
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
NEW, NEWC = "exp_E6_demo_hybrid", "#2f8f3a"
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


# ------------------------------------------------------------------------------------- tables
def t_training():
    L = ["| seed | epochs trained | stopping epoch | best-$J$ epoch | best $J$ | minutes |", "|---|---|---|---|---|---|"]
    for s in (101, 202, 303, 404, 505):
        m = rj(E6 / f"m{s}" / "summary.json")
        L.append(f"| {s} | {m['stopped_epoch'] + 1} | {m['stopped_epoch']} | {m['best_J_epoch']} | {m['best_J']:.5f} | "
                 f"{m['train_seconds'] / 60:.1f} |")
    return "\n".join(L) + "\n"


def t_sweep():
    sw = rj(E6 / "sweep_choice.json")
    L = ["| $w_{cd}$ | $w_{\\rm lcfs}$ | best $J$ (40 epochs) | best coefficient MSE | picked |", "|---|---|---|---|---|"]
    for r in sw["rows"]:
        L.append(f"| {r['w']['w_cd']:g} | {r['w']['w_lcfs']:.4g} | {r['best_J']:.5f} | {r['best_mse']:.3e} | "
                 f"{'yes' if r['i'] == sw['pick'] else ''} |")
    return "\n".join(L) + "\n"


def t_selection():
    s1, sel = rj(E6 / "s1.json"), rj(E6 / "select.json")
    L = ["| ensemble (archval100) | $J$ | $S$ | $R^2_\\psi$ | $R^2_{q_{95}}$ | $R^2_{\\beta_N}$ | $1-D_{\\rm LCFS}$ | Consistency |",
         "|---|---|---|---|---|---|---|---|"]

    def row(name, J, a):
        return (f"| {name} | {J:.5f} | {f6(a['S'])} | {f6(a['R2_psi'])} | {f6(a['R2_q95'])} | {f6(a['R2_betaN'])} | "
                f"{f6(a['one_minus_D_LCFS'])} | {f6(a['Consistency'])} |")
    L.append(row("E4 (3 seeds)", s1["E4_ensemble"]["J"], s1["E4_archval100"]))
    L.append(row("E3/E4 checkpoints re-selected by $J$ (3 seeds)", s1["S1_ensemble"]["J"], s1["S1_archval100"]))
    for r in sel["rows"]:
        tag = " (picked)" if r["k"] == sel["pick"] else ""
        L.append(row(f"E6 rank-{r['k']} states (5 seeds){tag}", r["objective"]["J"], r["archval100"]))
    return "\n".join(L) + "\n"


def t_scorebox(split, cols):
    hdr = "| row | " + " | ".join(c for c, _ in cols) + " |"
    L = [hdr, "|---|" + "---|" * len(cols)]
    for k, lab in ROWS + CONS:
        L.append(f"| {lab} | " + " | ".join(f6(s.get(k)) for _, s in cols) + " |")
    return "\n".join(L) + "\n"


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


# ------------------------------------------------------------------------------------ figures
def fig_ts(fr):
    for key, rows in fr.items():
        t = [num(r["t_ms"]) for r in rows]
        fig, axs = plt.subplots(3, 4, figsize=(10, 6.2))
        items = [(q, l, None, None) for q, l in TRACES] + list(METRICS)
        for ax, (q, l, f, ref) in zip(axs.ravel(), items):
            if f is None:
                ax.plot(t, [num(r[f"{q}_gt"]) for r in rows], "k-", lw=1.4, label="EFIT++")
                ax.plot(t, [num(r[f"{q}_pred"]) for r in rows], "-", color=NEWC, lw=1.2, label="E6")
            else:
                ax.axhline(ref * (f if q != "r2_psi_frame" else 1), color="k", lw=1.0)
                ax.plot(t, [None if num(r[q]) is None else num(r[q]) * f for r in rows], "-", color=NEWC, lw=1.2)
            ax.set_title(l)
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel("t (ms)")
        fig.suptitle(f"{key}: EFIT++ (black) and E6 (green)", fontsize=9)
        fig.tight_layout()
        save(fig, f"e6_ts_{key.replace('mast_shot_', '')}")


def fig_overlay(fr):
    from app.fec_site import app as site
    from pipeline.datasets import load_shot
    from pipeline import audit as A
    meta = rj(REPO / "outputs/experiments/preds" / f"{NEW}.json")
    z = np.load(REPO / "outputs/experiments/preds" / f"{NEW}.npz")
    R, Zg = A.R, A.Z
    for i, key in enumerate(meta["keys"]):
        s = load_shot("demo", key)
        gt = np.asarray(s.gt["psi"], float)
        pr = np.asarray(z[f"shot_{i:04d}_psirz"], float)
        tt = np.asarray(s.row["efit_times"], float).reshape(-1)
        T = gt.shape[0]
        ks = sorted({int(round(f * (T - 1))) for f in (0.2, 0.5, 0.9)})
        fig, axs = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 4.6))
        for ax, k in zip(np.atleast_1d(axs), ks):
            for psi, col, ls, lab in ((gt[k], "k", "-", "EFIT++"), (pr[k], NEWC, "--", "E6")):
                c = A.lcfs(psi)
                o = A.opoint(psi)
                xp = A.xpoints(psi)
                if c is not None:
                    ia = int(np.argmin(np.abs(R - o[0]))); ja = int(np.argmin(np.abs(Zg - o[1])))
                    pa = psi[ja, ia]
                    pb = float(np.mean([psi[int(np.argmin(np.abs(Zg - q[1]))), int(np.argmin(np.abs(R - q[0])))] for q in c]))
                    lev = sorted(pa + f * (pb - pa) for f in (0.2, 0.4, 0.6, 0.8))
                    ax.contour(R, Zg, psi, levels=lev, colors=col, linestyles=ls, linewidths=0.7, alpha=0.8)
                    ax.plot(c[:, 0], c[:, 1], color=col, ls=ls, lw=1.6, label=lab)
                ax.plot([o[0]], [o[1]], "o", mfc="none", mec=col, ms=7)
                for h in ("upper", "lower"):
                    if xp[h] is not None:
                        ax.plot([xp[h][0]], [xp[h][1]], "x", color=col, ms=8, mew=2)
            ax.set_aspect("equal")
            ax.set_xlim(0.1, 1.6); ax.set_ylim(-1.6, 1.6)
            ax.set_title(f"t = {tt[k]:.0f} ms")
            ax.set_xlabel("R (m)")
        np.atleast_1d(axs)[0].set_ylabel("Z (m)")
        np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{key}: LCFS, O/X-points and $\\psi_N$ = 0.2–0.8, EFIT++ (black) and E6 (green dashed)", fontsize=9)
        fig.tight_layout()
        save(fig, f"e6_overlay_{key.replace('mast_shot_', '')}")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------- internal check
def internal(cds_new):
    fin = rj(E6 / "final.json")
    L = ["# E6 internal check against the FEC Phase-2 submission (not part of the paper)", ""]
    for split in ("demo", "holdout"):
        c2, e4, e6 = fin[f"C2_{split}"], fin[f"E4_{split}"], fin[split]
        L += [f"## {split}", "", "| row | Phase-2 submission | E4 | E6 | E6 vs submission |", "|---|---|---|---|---|"]
        for k, lab in list(ROWS) + list(CONS):
            a, b, c = c2.get(k), e4.get(k), e6.get(k)
            w = "–" if a is None or c is None else ("E6 better" if c > a else ("tie" if c == a else "E6 worse"))
            L.append(f"| {PLAIN.get(k, k)} | {f6(a)} | {f6(b)} | {f6(c)} | {w} |")
        L.append("")
    fr_c2 = frames("demo_hybrid_shipped")
    cds_c2 = {k: curve_distance(v) for k, v in fr_c2.items()}
    fr_e4 = frames("exp_E4_demo_hybrid")
    cds_e4 = {k: curve_distance(v) for k, v in fr_e4.items()}
    L += ["## Curve distances on the three demo shots (E6 vs submission vs E4)", "",
          "| shot | curve | RMSE sub | RMSE E4 | RMSE E6 | area sub | area E4 | area E6 |", "|---|---|---|---|---|---|---|---|"]
    wins = {"rmse": 0, "area": 0, "rmse_e4": 0, "area_e4": 0, "n": 0}
    for key in sorted(cds_new):
        for q in cds_new[key]:
            a, b, c = cds_c2[key].get(q), cds_e4[key].get(q), cds_new[key].get(q)
            if not (a and b and c):
                continue
            wins["n"] += 1
            wins["rmse"] += c["rmse"] < a["rmse"]; wins["area"] += c["area"] < a["area"]
            wins["rmse_e4"] += b["rmse"] < a["rmse"]; wins["area_e4"] += b["area"] < a["area"]
            L.append(f"| {key[-5:]} | {q} | {a['rmse']:.4g} | {b['rmse']:.4g} | {c['rmse']:.4g} | {a['area']:.4g} | "
                     f"{b['area']:.4g} | {c['area']:.4g} |")
    L += ["", f"E6 closer than the submission: {wins['rmse']} / {wins['n']} by RMSE, {wins['area']} / {wins['n']} by area. "
          f"E4 closer than the submission: {wins['rmse_e4']} / {wins['n']} by RMSE, {wins['area_e4']} / {wins['n']} by area.", ""]
    (E6 / "internal_check.md").write_text("\n".join(L) + "\n")
    (E6 / "internal_wins.json").write_text(json.dumps(wins, indent=1) + "\n")
    return wins


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    fin = rj(E6 / "final.json")
    (GEN / "e6_training.md").write_text(t_training())
    (GEN / "e6_sweep.md").write_text(t_sweep())
    (GEN / "e6_selection.md").write_text(t_selection())
    (GEN / "e6_scorebox_demo.md").write_text(t_scorebox("demo", [("E6 (3 FEC demo shots)", fin["demo"])]))
    (GEN / "e6_holdout.md").write_text(t_scorebox("holdout", [("E4", fin["E4_holdout"]), ("E6", fin["holdout"])]))
    fr = frames(NEW)
    txt, cds = t_curves(fr)
    (GEN / "e6_curves_demo.md").write_text(txt)
    fig_ts(fr)
    fig_overlay(fr)
    sc = rj(E6 / "soft_check.json")["used"]
    L = ["| read-out | correlation of levels (EFIT++ $\\psi$) | correlation of errors (E4 $\\psi$ minus EFIT++) |", "|---|---|---|"]
    for k, lab in (("kappa", r"$\kappa$"), ("tri_top", r"$\delta_{\rm top}$"), ("tri_bot", r"$\delta_{\rm bot}$"),
                   ("volume", "$V$"), ("R_axis", r"$R_{\rm axis}$"), ("Z_axis", r"$Z_{\rm axis}$")):
        L.append(f"| {lab} | {sc[k]['r_level']:.3f} | {sc[k]['r_error']:.3f} |")
    (GEN / "e6_softcheck.md").write_text("\n".join(L) + "\n")
    wins = internal(cds)
    facts = {"demo": {k: fin["demo"].get(k) for k, _ in ROWS + CONS}, "holdout": {k: fin["holdout"].get(k) for k, _ in ROWS + CONS}}
    (GEN / "e6_facts.json").write_text(json.dumps(facts, indent=1, default=float) + "\n")
    print("ok", json.dumps(wins))


if __name__ == "__main__":
    main()
