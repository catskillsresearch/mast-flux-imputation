#!/usr/bin/env python3
"""Section 6 result tables, derived ONLY from saved audit summaries (``outputs/audit/<tag>/summary.json``).

Rows: OSF with the FAIR-MAST read-out maps (E5), the MSE-only hybrid (E1), the route-guided E4 and,
when its final stage has run, the curve-distance calibration E6. Output: ``paper/generated/res_*.md``.

    .venv/bin/python scripts/make_results_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUD = REPO / "outputs" / "audit"
GEN = REPO / "paper" / "generated"
MAIN = (("S", "$S$"), ("R2_psi", "$R^2_\\psi$"), ("R2_q95", "$R^2_{q_{95}}$"), ("R2_betaN", "$R^2_{\\beta_N}$"),
        ("one_minus_D_LCFS", "$1-D_{\\rm LCFS}$"), ("Consistency", "Cons"))
CONS = (("R2_R_axis", "$R_{\\rm axis}$"), ("R2_Z_axis", "$Z_{\\rm axis}$"), ("R2_kappa", "$\\kappa$"),
        ("R2_tri_top", "$\\delta_{\\rm top}$"), ("R2_tri_bot", "$\\delta_{\\rm bot}$"), ("R2_volume", "$V$"),
        ("R2_li", "$\\ell_i$"))
ROUTE = (("rate_jump", "tip gap $\\ge$ 50 mm", "pct"), ("p95_hausdorff_mm", "p95 $d_H$ (mm)", "mm"),
         ("mean_xpt_disp_mm", "X-pt (mm)", "mm"), ("rate_topology_flip", "flips", "pct"))


def summ(tag):
    p = AUD / tag / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else None


def f(v, kind="f"):
    if v is None:
        return "–"
    return f"{100 * v:.1f}%" if kind == "pct" else (f"{v:.1f}" if kind == "mm" else f"{v:.6f}")


def rows_for(split):
    r = [("OSF", f"exp_E5_{split}_osf"), ("E1 (MSE)", f"exp_E1_{split}_hybrid"),
         ("E4", f"exp_E4_{split}_hybrid"), ("E6", f"exp_E6_{split}_hybrid")]
    return [(lab, s) for lab, t in r if (s := summ(t))]


def write(split, name):
    rows = rows_for(split)
    m = ["| model | " + " | ".join(l for _, l in MAIN) + " | " + " | ".join(l for _, l, _ in ROUTE) + " |",
         "|---|" + "---|" * (len(MAIN) + len(ROUTE))]
    c = ["| model | " + " | ".join(l for _, l in CONS) + " |", "|---|" + "---|" * len(CONS)]
    for lab, s in rows:
        m.append(f"| {lab} | " + " | ".join(f(s.get(k)) for k, _ in MAIN) + " | "
                 + " | ".join(f(s.get(k), kd) for k, _, kd in ROUTE) + " |")
        c.append(f"| {lab} | " + " | ".join(f(s.get(k)) for k, _ in CONS) + " |")
    (GEN / f"res_{name}_main.md").write_text("\n".join(m) + "\n")
    (GEN / f"res_{name}_components.md").write_text("\n".join(c) + "\n")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    write("holdout", "holdout")
    print("wrote", *(p.name for p in sorted(GEN.glob("res_*.md"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
