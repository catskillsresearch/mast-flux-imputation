"""``mast_12_osf_0.0.12`` -- the MAST assignment of the osf_0.0.12 symbols.

The values are exactly those of catskills_fec ``osf/mast_12_osf_0_0_12.py``; this file only
splits them into four groups so it is clear what is hardware, what is a numerical gate, what
is an algorithm switch, and what was *calibrated*:

* ``MACHINE_SPEC``  -- fixed by MAST hardware / the FEC parquet schema (column names, TF turns,
  flux sign convention). Not tunable.
* ``GRID_GATES``    -- geometric search boxes and acceptance gates on the 65x65 MAST grid
  (where to look for the O-point / X-points, what counts as a compact closed contour).
* ``SWITCHES``      -- which code path osf_0.0.12 takes (how I_p, pressure and F_b are read).
* ``CALIBRATED``    -- the physics parameters set by the calibration (see docs/MODEL.md §OSF
  and DATA_PROVENANCE.md §4: physics end-members inspected on the 3 labelled
  starter-kit shots, plus 5 Codabench aggregate-score probes on mast_public_test).

``FINAL_SCALARS`` are the two post-hoc multipliers the shipped Phase-2 Final MAST submission
applied *outside* the solver (catskills_fec ``catskills/scripts/generate_final_predictions.py``):
``q95 <- clip(q95_osf * C_STAR / 1.20, 2, 25)`` and ``betaN <- clip(betaN_osf * S_BETA, 0.2, 5)``.
They are used by ``pipeline/predict.py --mode hybrid`` only (``--scalar-cal final``).
"""
from __future__ import annotations

from models.osf.solver import Calibration

MODEL_ID = "mast_12_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"

MACHINE_SPEC = dict(
    machine="MAST",
    n_tf=24.0,          # TF turns: F_vac = mu0 * N_TF * I_TF / (2 pi)
    sigma_ax=1.0,       # EFIT++/FAIR-MAST: magnetic axis is a psi MAXIMUM
    sol="magnetics_sol_current",
    v_u="magnetics_p2u_current",
    v_l="magnetics_p2l_current",
    delta=("magnetics_p4u_current", "magnetics_p4l_current"),
    outboard=("magnetics_p5u_current", "magnetics_p5l_current"),
    tf="magnetics_tf_current",
    ip="magnetics_plasma_current",
    ip_time="magnetics_time",
    loader="mast_columns",
)

GRID_GATES = dict(
    r_axis_min=0.65, r_axis_max=1.05, z_axis_max=0.50,   # O-point search box (m)
    kappa_max=2.6, z_span_max=2.8,                       # compact last-closed contour
    r_out_min=1.00, r_in_max=0.40, edge_cells=2.0,
    dz_x=3.2,                                            # reject X-point contours taller than this
    x_r_min=0.25, x_r_max=1.50, x_z_abs=0.40, x_dist=0.25,
    rt_lo=0.25, rt_hi=1.45,                              # Thomson R window for the iterate median
)

SWITCHES = dict(
    walk_on_measured=True,
    pressure_on_measured=True,
    beta_guess="osf_0_0_1",
    readout_pressure="lcfs",
    readout_ip="measured",
    axis_on_stored=False,
    axis_gate_z=True,
    fallback_closed=True,
    factor_fallback=False,
    smooth_inplace=False,
    post_clip_beta=False,
    fb_from_bt=True,
)

CALIBRATED = dict(
    # clocks: w = (1 - min(|I_P2|/|I_p| / r_star, 1))^2 ,  b = clip(I_sol / i_star_a, 0, 1)
    r_star=0.34,
    i_star_a=16.3e3,
    # current-support ellipse  R0 = r00 + rw w - rb b ;  Z0 = z00 - zb b ;  a_R = ar0 + aw w
    #                          a_Z = a_R + a0 + ab b + ad (1 - b)
    r00=0.82, rw=0.10, rb=0.02,
    z00=0.009, zb=0.004,
    ar0=0.50, aw=0.08,
    a0=0.08, ab=0.42, ad=0.18,
    # profiles: P'(tau) = 1 - tau ; FF'(tau) = -gamma_bar (1 - tau)
    gamma_bar=-1.0,
    # Ampere constraint  int j dA = k I_p ,  k = k0 + kw w + kb b + kd (1 - b)
    k0=0.70, kw=0.30, kb=0.15, kd=0.20,
    i0_a=0.0, i_b_a=0.0,
    # vacuum coil multipliers
    lam_v0=1.8, lam_vb=-1.0,     # P2 pair: lam_V = 1.8 - b
    lam_l=0.105,                 # extra on P2L: x (1 + 0.105 (1 - b))
    lam_delta=1.15,              # P4 pair
    lam_o=-0.18,                 # P5 pair: x (1 - 0.18 (1 - b))
    lam_s0=0.5, lam_sb=0.10,     # solenoid: x (0.5 + 0.1 b)
    # q95 read-out law on q_map = F_b W / 2 pi
    q_hi=17.0, q_lo=8.0, alpha_q=0.55, eta_q=0.35, zeta_q=2.0, p_q=3.0,
    # betaN read-out
    n_beta=7, eta_beta=0.20, beta_lo=0.2, beta_hi=4.0, beta_fb=1.3, s_beta=1.0,
)

CAL = Calibration(calibration_id=MODEL_ID, **MACHINE_SPEC, **GRID_GATES, **SWITCHES, **CALIBRATED)

# Post-hoc multipliers used by the shipped hybrid (M9 + OSF) Final submission. NB the
# generate_final_predictions.py docstring says "s* = 1.000000" but the code uses 1.078508.
FINAL_SCALARS = dict(c_star=1.062242, q_ref=1.20, s_beta=1.078508,
                     q_clip=(2.0, 25.0), beta_clip=(0.2, 5.0))


def apply_final_scalars(q95, beta_n):
    """The exact post-hoc scalar map of the shipped MAST Final submission."""
    import numpy as np

    f = FINAL_SCALARS
    q = np.clip(np.asarray(q95, np.float64) * (f["c_star"] / f["q_ref"]), *f["q_clip"])
    b = np.clip(np.asarray(beta_n, np.float64) * f["s_beta"], *f["beta_clip"])
    return q.astype(np.float32), b.astype(np.float32)
