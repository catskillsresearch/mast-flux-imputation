"""Component 1 -- OSF: one-shot forward Grad-Shafranov solver ``mast_12_osf_0.0.12``.

    from models.osf import solve_shot
    pack = solve_shot(row)      # row = one FEC-schema MAST parquet row (dict)
    pack["psirz"], pack["q95"], pack["betaN"]
"""
from __future__ import annotations

from models.osf.calibration import CAL, MODEL_ID, SOLVER_ID, apply_final_scalars
from models.osf.primitives import plasma_green
from models.osf.solver import solve_one_frame as _solve_one_frame
from models.osf.solver import solve_shot as _solve_shot


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    return _solve_one_frame(row, t_index, CAL, maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, CAL, green=green, maps=maps)


__all__ = ["CAL", "MODEL_ID", "SOLVER_ID", "solve_shot", "solve_one_frame", "plasma_green",
           "apply_final_scalars"]
