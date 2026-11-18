#!/usr/bin/env python3
"""Result figures and result tables for the paper, derived ONLY from saved run outputs.

Inputs (read-only): ``outputs/{holdout,demo}_{osf,hybrid}.{npz,json,score.json}`` written by
``pipeline.predict`` / ``pipeline.score``, plus the EFIT ground truth via ``pipeline.datasets``.
Outputs: ``paper/figures/res_*.{pdf,png}`` and ``paper/generated/*.md`` (tables @imported by the
Results notes). Nothing else is written.

    MPLCONFIGDIR=/tmp/mpl python scripts/make_result_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.fec_site import app as site  # noqa: E402  (read-only helpers: LCFS, O-point, predictions)
from pipeline.paths import output_dir  # noqa: E402

FIG = REPO / "paper" / "figures"
GEN = REPO / "paper" / "generated"
MODES = (("osf", "OSF"), ("hybrid", "M9 + OSF"))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})


def load_score(split, mode):
    return json.loads((output_dir() / f"{split}_{mode}.score.json").read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def f6(x):
    return f"{x:.6f}"


# ------------------------------------------------------------------ tables
def write_tables():
    rows_main, rows_comp, meta = {}, {}, {}
    for split in ("holdout", "demo"):
        main = ["| model | $S$ | $R^2_\\psi$ | $R^2_{q_{95}}$ | $R^2_{\\beta_N}$ | $1-D_{\\rm LCFS}$ | Consistency |",
                "|---|---|---|---|---|---|---|"]
        comp = ["| model | $R_{\\rm axis}$ | $Z_{\\rm axis}$ | $\\kappa$ | $\\delta_{\\rm top}$ | $\\delta_{\\rm bot}$ | $V$ | $\\ell_i$ |",
                "|---|---|---|---|---|---|---|---|"]
        for mode, label in MODES:
            s = load_score(split, mode)
            c, r = s["components"], s["result"]
            main.append(f"| {label} | {f6(c['S'])} | {f6(c['R2_psi'])} | {f6(c['R2_q95'])} | {f6(c['R2_betaN'])} "
                        f"| {f6(c['one_minus_D_LCFS'])} | {f6(c['Consistency'])} |")
            comp.append("| " + label + " | " + " | ".join(
                f6(c[k]) for k in ("R2_R_axis", "R2_Z_axis", "R2_kappa", "R2_tri_top", "R2_tri_bot",
                                    "R2_volume", "R2_li")) + " |")
            meta[(split, mode)] = r
        rows_main[split], rows_comp[split] = main, comp
        (GEN / f"{split}_main.md").write_text("\n".join(main) + "\n")
        (GEN / f"{split}_components.md").write_text("\n".join(comp) + "\n")
    # run facts
    lines = ["| split | model | shots | frames | $\\psi$ sign | LCFS fail | derive fail | per-shot mean $R^2_\\psi$ | scorer |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (split, mode), r in meta.items():
        lines.append(f"| {split} | {dict(MODES)[mode]} | {r['n_shots']} | {r['n_frames']} | {r['psi_sign']:+d} "
                     f"| {100 * r['lcfs_fail_frac']:.4f}% | {100 * r['cons_fail_frac']:.4f}% "
                     f"| {r['r2_psi_per_shot_mean']:.6f} | v{r['scoring_version']} |")
    (GEN / "run_facts.md").write_text("\n".join(lines) + "\n")
    # worst shots (holdout)
    ps = {mode: {d["key"]: d for d in load_score("holdout", mode)["per_shot"]} for mode, _ in MODES}
    keys = sorted(ps["hybrid"], key=lambda k: ps["hybrid"][k]["r2_psi"])[:8]
    w = ["| shot | frames | $R^2_\\psi$ OSF | $R^2_\\psi$ M9+OSF | $D_{\\rm LCFS}$ OSF | $D_{\\rm LCFS}$ M9+OSF |",
         "|---|---|---|---|---|---|"]
    for k in keys:
        a, b = ps["osf"][k], ps["hybrid"][k]
        w.append(f"| {k.replace('mast_shot_', '')} | {b['T']} | {a['r2_psi']:.4f} | {b['r2_psi']:.4f} "
                 f"| {a['dlcfs']:.4f} | {b['dlcfs']:.4f} |")
    (GEN / "holdout_worst.md").write_text("\n".join(w) + "\n")
    return ps


# ------------------------------------------------------------------ figures
def fig_pershot(ps):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = np.linspace(0.70, 1.0, 31)
    for mode, label in MODES:
        r2 = np.array([d["r2_psi"] for d in ps[mode].values()])
        dl = np.array([d["dlcfs"] for d in ps[mode].values()])
        ax[0].hist(np.clip(r2, bins[0], 1), bins=bins, alpha=0.6, label=label)
        ax[1].hist(dl, bins=np.linspace(0, 0.6, 31), alpha=0.6, label=label)
    ax[0].set_xlabel(r"per-shot $R^2_\psi$ (clipped at 0.70)")
    ax[1].set_xlabel(r"per-shot mean $D_{\rm LCFS}$")
    for a in ax:
        a.set_ylabel("shots")
        a.legend(frameon=False)
    fig.suptitle("Sealed-80 holdout, 65 shots with diverted frames", y=1.02)
    save(fig, "res_pershot_hist")


def pick_shot(ps):
    r2 = {k: d["r2_psi"] for k, d in ps["hybrid"].items() if d["T"] >= 40}
    ks = sorted(r2, key=r2.get)
    return ks[len(ks) // 2]


def fig_psi(key):
    shot = site._shot("holdout", key)
    gt = shot.gt["psi"]
    k = gt.shape[0] // 2
    t = float(np.asarray(shot.row["efit_times"])[k])
    maps = {"EFIT++": gt[k]}
    for mode, label in MODES:
        maps[label] = site._pred("holdout", mode, key)["psi"][k]
    lc_ref = site._lcfs(maps["EFIT++"])
    lo, hi = np.nanpercentile(maps["EFIT++"][site.MASK], [2, 99.5])
    levels = np.linspace(lo, hi, 21)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4), sharey=True)
    for ax, (name, p) in zip(axes, maps.items()):
        pm = np.where(site.MASK, p, np.nan)
        ax.contour(site.R, site.Z, pm, levels=levels, linewidths=0.5, cmap="viridis")
        lc = site._lcfs(p)
        if lc_ref is not None and name != "EFIT++":
            ax.plot(lc_ref[:, 0], lc_ref[:, 1], "k--", lw=1.0, label="EFIT++ LCFS")
        if lc is not None:
            ax.plot(lc[:, 0], lc[:, 1], "r-", lw=1.2, label=f"{name} LCFS")
        ro, zo = site._opoint(p)
        if ro is not None:
            ax.plot([ro], [zo], "r+", ms=8)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 2.0)
        ax.set_title(name)
        ax.set_xlabel("R (m)")
        ax.legend(loc="lower right", fontsize=6, frameon=False)
    axes[0].set_ylabel("Z (m)")
    fig.suptitle(f"MAST shot {key.replace('mast_shot_', '')}, t = {t:.1f} ms (holdout frame {k + 1}/{gt.shape[0]})")
    save(fig, "res_psi_compare")
    return t, k


def fig_traces(key):
    shot = site._shot("holdout", key)
    t = np.asarray(shot.row["efit_times"], float)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
    for i, (q, lab) in enumerate((("q95", r"$q_{95}$"), ("betaN", r"$\beta_N$"))):
        ax[i].plot(t, shot.gt[q], "k-", lw=1.4, label="EFIT++")
        for mode, label in MODES:
            ax[i].plot(t, site._pred("holdout", mode, key)[q], lw=1.0, label=label)
        ax[i].set_ylabel(lab)
    ax[0].legend(frameon=False, ncol=3, fontsize=7)
    ax[1].set_xlabel("t (ms)")
    fig.suptitle(f"MAST shot {key.replace('mast_shot_', '')}: scalar read-outs vs EFIT++ (diverted frames)")
    save(fig, "res_scalar_traces")


def fig_scatter():
    meta = json.loads((output_dir() / "holdout_hybrid.json").read_text())
    z = np.load(output_dir() / "holdout_hybrid.npz")
    zq = {"osf": np.load(output_dir() / "holdout_osf.npz")}
    gq, gb, pq, pb, oq, ob = [], [], [], [], [], []
    keys_osf = json.loads((output_dir() / "holdout_osf.json").read_text())["keys"]
    for i, key in enumerate(meta["keys"]):
        s = site._shot("holdout", key)
        j = keys_osf.index(key)
        gq.append(s.gt["q95"]); gb.append(s.gt["betaN"])
        pq.append(z[f"shot_{i:04d}_q95"]); pb.append(z[f"shot_{i:04d}_betaN"])
        oq.append(zq["osf"][f"shot_{j:04d}_q95"]); ob.append(zq["osf"][f"shot_{j:04d}_betaN"])
    gq, gb, pq, pb, oq, ob = map(np.concatenate, (gq, gb, pq, pb, oq, ob))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for a, g, p, o, lab, lim in ((ax[0], gq, pq, oq, r"$q_{95}$", (2, 16)), (ax[1], gb, pb, ob, r"$\beta_N$", (0, 6))):
        a.scatter(g, o, s=2, alpha=0.25, label="OSF")
        a.scatter(g, p, s=2, alpha=0.25, label="M9 + OSF")
        a.plot(lim, lim, "k-", lw=0.8)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"EFIT++ {lab}"); a.set_ylabel(f"predicted {lab}")
        a.legend(frameon=False, markerscale=4)
    fig.suptitle(f"Sealed-80 holdout, all {len(gq)} diverted frames")
    save(fig, "res_scalar_scatter")
    return len(gq)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    ps = write_tables()
    fig_pershot(ps)
    key = pick_shot(ps)
    t, k = fig_psi(key)
    fig_traces(key)
    n = fig_scatter()
    info = {"example_shot": key, "example_frame_index": k, "example_time_ms": t, "scatter_frames": n,
            "example_r2_psi": {m: ps[m][key]["r2_psi"] for m, _ in MODES}}
    (GEN / "figure_facts.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
