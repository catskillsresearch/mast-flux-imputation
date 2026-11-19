#!/usr/bin/env python3
"""Audit tables for the paper, derived ONLY from saved ``pipeline.audit`` summaries.

Inputs (read-only): ``outputs/audit/exp_E1_{demo,holdout}_hybrid`` (MSE-only hybrid) and ``exp_E5_{demo,holdout}_osf`` written by
``python -m experiments.driver``. Output: ``paper/generated/audit_*.md`` (@imported by the paper).

    .venv/bin/python scripts/make_audit_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.paths import output_dir  # noqa: E402

GEN = REPO / "paper" / "generated"
RUNS = [("demo", "hybrid"), ("holdout", "hybrid"), ("demo", "osf"), ("holdout", "osf")]
HEAD = {"hybrid": "M9 + OSF (E1)", "osf": "OSF"}
TAG = {"hybrid": "exp_E1_{s}_hybrid", "osf": "exp_E5_{s}_osf"}

# (key, label, kind) kind: f = 6 decimals, mm = 1 decimal, pct = percent with 1 decimal
TARGET = [("S", "$S$", "f"), ("R2_psi", "$R^2_\\psi$", "f"), ("R2_q95", "$R^2_{q_{95}}$", "f"),
          ("R2_betaN", "$R^2_{\\beta_N}$", "f"), ("one_minus_D_LCFS", "$1-D_{\\rm LCFS}$", "f"),
          ("Consistency", "Cons", "f"), ("R2_Z_axis", "$R^2(Z_{\\rm axis})$", "f"),
          ("R2_tri_bot", "$R^2(\\delta_{\\rm bot})$", "f"), ("R2_li", "$R^2(\\ell_i)$", "f")]
ROUTE = [("mean_hausdorff_mm", "mean LCFS Hausdorff (mm)", "mm"),
         ("p95_hausdorff_mm", "p95 LCFS Hausdorff (mm)", "mm"),
         ("mean_area_symdiff_rel", "mean LCFS area mismatch", "pct"),
         ("mean_opoint_disp_mm", "mean O-point offset (mm)", "mm"),
         ("mean_xpt_disp_mm", "mean X-point offset (mm)", "mm"),
         ("rate_jump", "frames with LCFS tip gap $\\ge$ 50 mm", "pct"),
         ("rate_xpt_lost", "frames with X-point lost", "pct"),
         ("rate_topology_flip", "limited/diverted flips", "pct"),
         ("mae_q95", "mean $\\lvert\\Delta q_{95}\\rvert$", "f"), ("mae_betaN", "mean $\\lvert\\Delta\\beta_N\\rvert$", "f"),
         ("mean_abs_lag", "mean absolute lag of scalar traces (frames)", "f")]


def fmt(v, kind):
    if kind == "mm":
        return f"{v:.1f}"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    return f"{v:.6f}"


def table(rows, runs, data):
    cols = [f"{HEAD[m]} {s}" for s, m in runs]
    out = ["| quantity | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, lab, kind in rows:
        out.append(f"| {lab} | " + " | ".join(fmt(data[r][k], kind) for r in runs) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    data = {}
    for s, m in RUNS:
        p = output_dir() / "audit" / TAG[m].format(s=s) / "summary.json"
        data[(s, m)] = json.loads(p.read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "audit_target.md").write_text(table(TARGET, RUNS, data))
    (GEN / "audit_route.md").write_text(table(ROUTE, RUNS, data))
    n = {r: (data[r]["n_shots"], data[r]["n_frames"]) for r in RUNS}
    (GEN / "audit_facts.json").write_text(json.dumps({f"{m}_{s}": v for (s, m), v in n.items()}) + "\n")
    print("wrote", *(p.name for p in sorted(GEN.glob("audit_*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
