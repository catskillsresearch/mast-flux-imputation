#!/usr/bin/env python3
"""Paper tables and figures for the model: ground truth (EFIT++) vs model prediction only.

Every number is read from saved outputs; nothing is typed in.

Inputs
  outputs/compare/demo_e6.npz (+ .json)       model predictions on the three validation shots
                                              (FAIR-MAST rows, ``pipeline.datasets`` split ``demo``)
  outputs/audit/compare_demo_e6/              per-frame audit of those predictions
  outputs/audit/exp_E6_holdout_hybrid/        audit of the sealed holdout (65 shots)
  outputs/experiments/E6/                     training records (m*/summary.json, sweep_choice.json,
                                              select.json, soft_check.json)
Outputs
  paper/generated/model_*.md, model_facts.json; paper/figures/model_{ts,overlay}_<shot>.{pdf,png}

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/make_model_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import make_e6_report as R  # noqa: E402  (shared curve-distance code and constants)

plt = R.plt
RUN = REPO / "outputs" / "experiments" / "E6"
AUD, GEN, FIG = R.AUD, R.GEN, R.FIG
PRED = REPO / "outputs" / "compare" / "demo_e6.npz"
VAL_TAG, HOLD_TAG = "compare_demo_e6", "exp_E6_holdout_hybrid"
C = R.NEWC
rj, f6 = R.rj, R.f6


def per_shot_scores():
    from pipeline.datasets import load_shot
    from pipeline.score import components, load_predictions, score_fold
    meta = rj(PRED.with_suffix(".json"))
    preds = load_predictions(PRED, len(meta["keys"]))
    out = {}
    for k, p in zip(meta["keys"], preds):
        s = load_shot("demo", k)
        res, _ = score_fold([s.gt], [p], workers=1)
        out[k] = components(res)
    return out


def t_scorebox(cols):
    L = ["| row | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, lab in R.ROWS + R.CONS:
        L.append(f"| {lab} | " + " | ".join(f6(s.get(k)) for _, s in cols) + " |")
    return "\n".join(L) + "\n"


ROUTE = (("rate_jump", "frames with LCFS tip gap $\\ge$ 50 mm", "pct"), ("mean_hausdorff_mm", "mean LCFS Hausdorff (mm)", "mm"),
         ("p95_hausdorff_mm", "p95 LCFS Hausdorff (mm)", "mm"), ("mean_xpt_disp_mm", "mean X-point offset (mm)", "mm"),
         ("mean_opoint_disp_mm", "mean O-point offset (mm)", "mm"), ("rate_topology_flip", "limited/diverted flips", "pct"))


def fmt(v, kind):
    if v is None:
        return "–"
    return f"{100 * v:.1f}%" if kind == "pct" else f"{v:.1f}"


def t_route(cols):
    L = ["| quantity | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, lab, kind in ROUTE:
        L.append(f"| {lab} | " + " | ".join(fmt(s.get(k), kind) for _, s in cols) + " |")
    return "\n".join(L) + "\n"


def t_training():
    L = ["| seed | epochs trained | best-$J$ epoch | best $J$ | returns to best weights | minutes |", "|---|---|---|---|---|---|"]
    for s in (101, 202, 303, 404, 505):
        m = rj(RUN / f"m{s}" / "summary.json")
        L.append(f"| {s} | {m['stopped_epoch'] + 1} | {m['best_J_epoch']} | {m['best_J']:.5f} | {len(m.get('rollbacks') or [])} | "
                 f"{m['train_seconds'] / 60:.1f} |")
    return "\n".join(L) + "\n"


def t_sweep():
    sw = rj(RUN / "sweep_choice.json")
    L = ["| $w_{cd}$ | best $J$ (40 epochs, seed 101) | chosen |", "|---|---|---|"]
    for r in sw["rows"]:
        L.append(f"| {r['w']['w_cd']:g} | {r['best_J']:.5f} | {'yes' if r['i'] == sw['pick'] else ''} |")
    return "\n".join(L) + "\n"


def t_selection():
    sel = rj(RUN / "select.json")
    L = ["| ensemble of the five seeds | $J$ | $S$ (archive validation) | used |", "|---|---|---|---|"]
    for r in sel["rows"]:
        L.append(f"| rank-{r['k']} states | {r['objective']['J']:.5f} | {f6(r['archval100']['S'])} | "
                 f"{'yes' if r['k'] == sel['pick'] else ''} |")
    return "\n".join(L) + "\n", sel


def t_soft():
    sc = rj(RUN / "soft_check.json")["used"]
    L = ["| read-out | correlation of values on EFIT++ $\\psi$ | correlation of errors on network $\\psi$ |", "|---|---|---|"]
    for k, lab in (("kappa", r"$\kappa$"), ("tri_top", r"$\delta_{\rm top}$"), ("tri_bot", r"$\delta_{\rm bot}$"),
                   ("volume", "$V$"), ("R_axis", r"$R_{\rm axis}$"), ("Z_axis", r"$Z_{\rm axis}$")):
        L.append(f"| {lab} | {sc[k]['r_level']:.3f} | {sc[k]['r_error']:.3f} |")
    return "\n".join(L) + "\n"


def betan_tail(fr):
    out = {}
    for k, rows in fr.items():
        e = [(R.num(r["betaN_pred"]) - R.num(r["betaN_gt"])) ** 2 for r in rows]
        t = [R.num(r["t_ms"]) for r in rows]
        out[k] = {"share_last5": sum(e[-5:]) / sum(e), "t_last5_ms": t[-5], "rmse_all": (sum(e) / len(e)) ** 0.5,
                  "rmse_without_last5": (sum(e[:-5]) / (len(e) - 5)) ** 0.5}
    return out


def fig_ts(fr):
    for key, rows in fr.items():
        t = [R.num(r["t_ms"]) for r in rows]
        fig, axs = plt.subplots(3, 4, figsize=(10, 6.2))
        items = [(q, l, None, None) for q, l in R.TRACES] + list(R.METRICS)
        for ax, (q, l, f, ref) in zip(axs.ravel(), items):
            if f is None:
                ax.plot(t, [R.num(r[f"{q}_gt"]) for r in rows], "k-", lw=1.4, label="EFIT++")
                ax.plot(t, [R.num(r[f"{q}_pred"]) for r in rows], "-", color=C, lw=1.2, label="model")
            else:
                ax.axhline(ref * (f if q != "r2_psi_frame" else 1), color="k", lw=1.0)
                ax.plot(t, [None if R.num(r[q]) is None else R.num(r[q]) * f for r in rows], "-", color=C, lw=1.2)
            ax.set_title(l)
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel("t (ms)")
        fig.suptitle(f"{key}: EFIT++ (black) and model (green)", fontsize=9)
        fig.tight_layout()
        R.save(fig, f"model_ts_{key.replace('mast_shot_', '')}")


def fig_overlay():
    from pipeline.datasets import load_shot
    from pipeline import audit as A
    meta = rj(PRED.with_suffix(".json"))
    z = np.load(PRED)
    Rg, Zg = A.R, A.Z
    for i, key in enumerate(meta["keys"]):
        s = load_shot("demo", key)
        gt = np.asarray(s.gt["psi"], float)
        pr = np.asarray(z[f"shot_{i:04d}_psirz"], float)
        tt = np.asarray(s.row["efit_times"], float).reshape(-1)
        T = gt.shape[0]
        ks = sorted({int(round(f * (T - 1))) for f in (0.2, 0.5, 0.9)})
        fig, axs = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 4.6))
        for ax, k in zip(np.atleast_1d(axs), ks):
            for psi, col, ls, lab in ((gt[k], "k", "-", "EFIT++"), (pr[k], C, "--", "model")):
                c = A.lcfs(psi)
                o = A.opoint(psi)
                xp = A.xpoints(psi)
                if c is not None:
                    ia = int(np.argmin(np.abs(Rg - o[0]))); ja = int(np.argmin(np.abs(Zg - o[1])))
                    pa = psi[ja, ia]
                    pb = float(np.mean([psi[int(np.argmin(np.abs(Zg - q[1]))), int(np.argmin(np.abs(Rg - q[0])))] for q in c]))
                    lev = sorted(pa + f * (pb - pa) for f in (0.2, 0.4, 0.6, 0.8))
                    ax.contour(Rg, Zg, psi, levels=lev, colors=col, linestyles=ls, linewidths=0.7, alpha=0.8)
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
        fig.suptitle(f"{key}: LCFS, O/X-points and $\\psi_N$ = 0.2–0.8, EFIT++ (black) and model (green dashed)", fontsize=9)
        fig.tight_layout()
        R.save(fig, f"model_overlay_{key.replace('mast_shot_', '')}")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    val = rj(AUD / VAL_TAG / "summary.json")
    hold = rj(AUD / HOLD_TAG / "summary.json")
    ps = per_shot_scores()
    cols = [(k.replace("mast_shot_", ""), v) for k, v in sorted(ps.items())]
    cols += [("3 validation shots", val), ("sealed holdout (65 shots)", hold)]
    (GEN / "model_scorebox.md").write_text(t_scorebox(cols))
    (GEN / "model_route.md").write_text(t_route([("3 validation shots", val), ("sealed holdout (65 shots)", hold)]))
    fr = R.frames(VAL_TAG)
    txt, cds = R.t_curves(fr)
    (GEN / "model_curves.md").write_text(txt)
    (GEN / "model_training.md").write_text(t_training())
    (GEN / "model_sweep.md").write_text(t_sweep())
    st, sel = t_selection()
    (GEN / "model_selection.md").write_text(st)
    (GEN / "model_softcheck.md").write_text(t_soft())
    fig_ts(fr)
    fig_overlay()
    keys = ("S", "R2_psi", "R2_q95", "R2_betaN", "one_minus_D_LCFS", "Consistency", "R2_R_axis", "R2_Z_axis", "R2_kappa",
            "R2_tri_top", "R2_tri_bot", "R2_volume", "R2_li") + tuple(k for k, _, _ in ROUTE)
    facts = {"per_shot": ps, "validation": {k: val.get(k) for k in keys}, "holdout": {k: hold.get(k) for k in keys},
             "betaN_tail": betan_tail(fr), "frames": {k: len(v) for k, v in fr.items()},
             "selection": {"pick": sel["pick"], "rows": [{"k": r["k"], "J": r["objective"]["J"], "S": r["archval100"]["S"]} for r in sel["rows"]],
                           "gate_S": sel.get("gate_S")}}
    (GEN / "model_facts.json").write_text(json.dumps(facts, indent=1, default=float) + "\n")
    print(json.dumps(facts, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
