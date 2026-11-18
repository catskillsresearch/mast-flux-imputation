# OSF

Free-boundary Grad–Shafranov forward solver with a measured-current-driven plasma model.

* `primitives.py`: Green's functions, coil unit maps, flux-surface tools.
* `solver.py`: the generic `osf_0.0.12` solver (`Calibration` dataclass, `solve_shot`).
* `calibration.py`: the MAST values, split into `MACHINE_SPEC`, `GRID_GATES`, `SWITCHES`,
  `CALIBRATED` (38) and the hybrid `FINAL_SCALARS`.

`from models.osf import solve_shot; pack = solve_shot(row)` → `psirz`, `q95`, `betaN`.
Equations: docs/MODEL.md.
