#!/usr/bin/env python3
"""Experiment tables for the paper, derived ONLY from saved audit summaries and experiment JSON.

Inputs (read-only): ``outputs/audit/<tag>/summary.json`` and ``outputs/experiments/*/{result,members,
sweep_choice,fit}.json`` written by ``python -m experiments.driver``. Output: ``paper/generated/exp_*.md``.

    .venv/bin/python scripts/make_experiment_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.common import route_score  # noqa: E402

AUD = REPO / "outputs" / "audit"
EXP = REPO / "outputs" / "experiments"
GEN = REPO / "paper" / "generated"

COLS = [("S", "$S$", "f"), ("R2_psi", "$R^2_\\psi$", "f"), ("R2_q95", "$R^2_{q_{95}}$", "f"),
        ("R2_betaN", "$R^2_{\\beta_N}$", "f"), ("one_minus_D_LCFS", "$1-D$", "f"), ("Consistency", "Cons", "f"),
        ("rate_jump", "tip gap $\\ge$50 mm", "pct"), ("p95_hausdorff_mm", "p95 $d_H$ (mm)", "mm"),
        ("mean_xpt_disp_mm", "X-pt (mm)", "mm"), ("rate_topology_flip", "flips", "pct")]


def rj(p):
    return json.loads(Path(p).read_text())


def summ(tag):
    p = AUD / tag / "summary.json"
    return rj(p) if p.is_file() else None


def fmt(v, kind):
    if v is None:
        return "–"
    if kind == "mm":
        return f"{v:.1f}"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "r":
        return f"{v:.4f}"
    return f"{v:.4f}"


def table(rows, ref):
    head = "| model | " + " | ".join(c[1] for c in COLS) + " | $\\mathcal R$ |"
    out = [head, "|---|" + "---|" * (len(COLS) + 1)]
    for lab, tag in rows:
        s = summ(tag)
        if s is None:
            continue
        r = route_score(s, ref)
        out.append(f"| {lab} | " + " | ".join(fmt(s.get(k), kind) for k, _, kind in COLS) + f" | {r:.4f} |")
    return "\n".join(out) + "\n"


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    ex = [E for E in ("E1", "E2", "E3", "E4") if (EXP / E / "result.json").is_file()]
    names = {"E1": "E1 (MSE)", "E2": "E2 (+band, LCFS)", "E3": "E2 + scalar heads (E3)", "E4": "E3 + crit., temporal (E4)"}
    for split, old in (("demo", "demo_hybrid_shipped"), ("archval100", "exp_archval100_hybrid_shipped")):
        rows = [("shipped M9 + OSF", old)] + [(names[E], f"exp_{E}_{split}_hybrid") for E in ex]
        (GEN / f"exp_{split}.md").write_text(table(rows, summ(old)))
    rows = [("shipped OSF", "demo_osf_shipped"), ("OSF + E5 maps", "exp_E5_demo_osf"),
            ("shipped OSF, archval100", "exp_archval100_osf_shipped"), ("OSF + E5 maps, archval100", "exp_E5_archval100_osf")]
    out = ["| run | $S$ | $R^2_{q_{95}}$ | $R^2_{\\beta_N}$ | mean $\\lvert\\Delta q_{95}\\rvert$ | mean $\\lvert\\Delta\\beta_N\\rvert$ |",
           "|---|---|---|---|---|---|"]
    for lab, tag in rows + [("shipped OSF, holdout", "holdout_osf_shipped"), ("OSF + E5 maps, holdout", "exp_E5_holdout_osf")]:
        s = summ(tag)
        if s:
            out.append(f"| {lab} | {s['S']:.4f} | {s['R2_q95']:.4f} | {s['R2_betaN']:.4f} | {s['mae_q95']:.4f} | {s['mae_betaN']:.4f} |")
    (GEN / "exp_e5.md").write_text("\n".join(out) + "\n")
    # sweeps
    out = ["| stage | setting | $\\mathcal R$ (select30) | $R^2_\\psi$ | $R^2_{qb}$ | chosen |", "|---|---|---|---|---|---|"]
    for E in ("E2", "E3", "E4"):
        p = EXP / E / "sweep_choice.json"
        if not p.is_file():
            continue
        d = rj(p)
        for r in d["rows"]:
            w = ", ".join(f"{k}={v:g}" for k, v in r["w"].items())
            out.append(f"| {E} | {w} | {r['R']:.4f} | {r['R2_psi']:.6f} | {r['R2_qb']:.6f} | {'yes' if r['i'] == d['pick'] else ''} |")
    (GEN / "exp_sweeps.md").write_text("\n".join(out) + "\n")
    # members
    out = ["| stage | seed | chosen state | $\\mathcal R$ (select30) | best-MSE epoch |", "|---|---|---|---|---|"]
    for E in ("E1", "E2", "E3", "E4"):
        p = EXP / E / "members.json"
        if not p.is_file():
            continue
        for s in rj(p)["selection"]:
            rr = {r["label"]: r["R"] for r in s.get("rows", [])}
            ep = s.get("best_mse_epoch", s.get("epoch"))
            out.append(f"| {E} | {s['seed']} | {s['pick']} | {fmt(rr.get(s['pick']), 'r')} | {ep} |")
    (GEN / "exp_members.md").write_text("\n".join(out) + "\n")
    # holdout
    hp = EXP / "holdout" / "result.json"
    if hp.is_file():
        pick = rj(hp)["pick"]
        rows = [("shipped M9 + OSF", "holdout_hybrid_shipped"), (names["E1"], "exp_E1_holdout_hybrid")]
        if pick != "E1":
            rows.append((names[pick] + " (pick)", f"exp_{pick}_holdout_hybrid"))
        (GEN / "exp_holdout.md").write_text(table(rows, summ("holdout_hybrid_shipped")))
    print("wrote", *(p.name for p in sorted(GEN.glob("exp_*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
