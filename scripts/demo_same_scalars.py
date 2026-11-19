#!/usr/bin/env python3
"""Diagnostic (not used for any selection): experiment flux ensembles with the shipped OSF scalar
read-out (c*, s*) on the three validation shots, so that the comparison with the shipped hybrid
differs only in psi. Writes outputs/audit/exp_<E>_demo_psi_shipscal/.

    .venv/bin/python scripts/demo_same_scalars.py E1 E2 E3 E4
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.driver import pred_audit  # noqa: E402

for E in sys.argv[1:]:
    files = json.loads((REPO / "outputs/experiments" / E / "members.json").read_text())["files"]
    s = pred_audit("demo", f"exp_{E}_demo_psi_shipscal", members=[Path(f) for f in files], scalars="osf_shipped")
    print(E, "S", s["S"], "R2_psi", s["R2_psi"], "R2_q95", s["R2_q95"])
