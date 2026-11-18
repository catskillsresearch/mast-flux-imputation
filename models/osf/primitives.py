"""Primitives of the MAST one-shot-forward solver (from catskills_fec ``osf/osf_0_0_1.py``).

Copied verbatim except that the osf_0.0.1 driver (its own ``solve_frame`` / ``solve_shot`` /
``read_q95_betan`` and helpers) is removed: ``osf_0.0.12`` (``solver.py``) only imports the
pieces kept here. NOTE: ``knobs_for_frame`` / ``first_q95`` / ``first_beta_n`` read THIS module's
first-guess constants (R0_M = 0.85 m, A_TRIAL_M = 0.55 m), not the osf_0.0.12 ellipse -- that is
how the shipped solver behaved and is preserved deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from scipy.special import ellipe, ellipk

from models.io import interpolate

MODEL_ID = "osf_0.0.1"

MU0 = 4.0e-7 * np.pi
E_CHARGE = 1.602176634e-19

# fec.md § osf_0.0.1
N_TF = 24.0
R0_M = 0.85
A_TRIAL_M = 0.55
Z0_M = 0.0
N_F = 2
N_P = 2
C_MODEL = (0.0, 0.0)
GAMMA_BAR = 0.0
D_MODEL = (0.0, 0.0)
Q95_CLIP = (2.0, 16.0)
BETAN_CLIP = (0.2, 4.0)
BETAN_TYPICAL = 1.3
PSI_ABS_TOL = 1.0e-4
SCALAR_RTOL = 1.0e-3
MAX_ITERS = 20
TAU_95 = 0.95
P_COIL_PREFIX = "magnetics_p"

P_COIL_COLUMNS = (
    "magnetics_p2l_current",
    "magnetics_p2u_current",
    "magnetics_p3l_current",
    "magnetics_p3u_current",
    "magnetics_p4l_current",
    "magnetics_p4u_current",
    "magnetics_p5l_current",
    "magnetics_p5u_current",
    "magnetics_p6l_current",
    "magnetics_p6u_current",
)
SOL_COLUMN = "magnetics_sol_current"
TF_COLUMN = "magnetics_tf_current"
IP_COLUMN = "magnetics_plasma_current"
IGNORED_IN_GREEN = (TF_COLUMN, "magnetics_efps_current")


@dataclass(frozen=True)
class FrameResult:
    psi: np.ndarray
    q95: float
    beta_n: float
    n_iter: int
    axis_r: float
    axis_z: float
    w: float
    delta_psi: float


def _as1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _interp_col(row: dict, column: str, t_dst: np.ndarray) -> np.ndarray:
    t_src = _as1(row["magnetics_time"])
    y = interpolate(t_src, row[column], t_dst).reshape(-1)
    if np.any(~np.isfinite(y)):
        fill = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
        y = np.nan_to_num(y, nan=fill)
    return y


def green_filament(r: np.ndarray, z: np.ndarray, rc: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """G(R,Z; Rc, Zc) in Wb/rad per ampere. Page: k^2, K(k^2), E(k^2)."""
    rt = np.asarray(r, dtype=np.float64)
    zt = np.asarray(z, dtype=np.float64)
    rs = np.asarray(rc, dtype=np.float64)
    zs = np.asarray(zc, dtype=np.float64)
    den = (rt + rs) ** 2 + (zt - zs) ** 2 + 1.0e-12
    k2 = np.clip(4.0 * rt * rs / den, 0.0, 1.0 - 1.0e-12)
    k = np.sqrt(np.maximum(k2, 1.0e-30))
    bracket = (2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)
    g = MU0 * np.sqrt(np.maximum(rt * rs, 0.0)) * bracket / (2.0 * np.pi * k)
    return np.where(rt == 0.0, 0.0, g)


def _rect_offsets(width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """3×3 cell centres: offsets ±w/3, ±h/3."""
    u = np.array([-1.0, 0.0, 1.0]) * float(width) / 3.0
    v = np.array([-1.0, 0.0, 1.0]) * float(height) / 3.0
    du, dv = np.meshgrid(u, v, indexing="xy")
    return du.ravel(), dv.ravel()


def coil_unit_maps(row: dict) -> dict[str, np.ndarray]:
    """One Wb/rad-per-ampere-turn map per current column (TF/efps omitted)."""
    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    cols = np.asarray(row["coil_input_column"], dtype=str).reshape(-1)
    cr = np.asarray(row["coil_R"], dtype=np.float64).reshape(-1)
    cz = np.asarray(row["coil_Z"], dtype=np.float64).reshape(-1)
    cw = np.asarray(row["coil_width"], dtype=np.float64).reshape(-1)
    ch = np.asarray(row["coil_height"], dtype=np.float64).reshape(-1)
    maps: dict[str, np.ndarray] = {}
    for column in list(dict.fromkeys(cols.tolist())):
        if column in IGNORED_IN_GREEN:
            continue
        idx = np.flatnonzero(cols == column)
        n_share = int(idx.size)
        # P-coils: kA·turn, each element carries I/n_share. Sol: plain kA, each row is a turn.
        element_divisor = n_share if column.startswith(P_COIL_PREFIX) else 1.0
        acc = np.zeros_like(rr)
        for i in idx:
            du, dv = _rect_offsets(float(cw[i]), float(ch[i]))
            elem = np.zeros_like(rr)
            for d_r, d_z in zip(du, dv, strict=True):
                elem += green_filament(rr, zz, cr[i] + d_r, cz[i] + d_z)
            acc += elem / 9.0 / element_divisor
        maps[column] = acc
    return maps


def vacuum_psi(maps: dict[str, np.ndarray], currents_a: dict[str, float]) -> np.ndarray:
    acc = None
    for column, basis in maps.items():
        piece = basis * float(currents_a.get(column, 0.0))
        acc = piece if acc is None else acc + piece
    return acc if acc is not None else np.zeros((65, 65))


def plasma_green(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(N,N) map: j_φ [A/m^2] → ψ_pl [Wb/rad], including dR dZ."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    rt = rr.ravel()[:, None]
    zt = zz.ravel()[:, None]
    rs = rr.ravel()[None, :]
    zs = zz.ravel()[None, :]
    g = green_filament(rt, zt, rs, zs)
    d_r = float(np.mean(np.diff(r)))
    d_z = float(np.mean(np.diff(z)))
    return g * abs(d_r * d_z)


def _contains(poly: np.ndarray, x: float, y: float) -> bool:
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crossing = ((y0 > y) != (y1 > y)) & (
        x < (x1 - x0) * (y - y0) / (y1 - y0 + 1.0e-30) + x0
    )
    return bool(crossing.sum() % 2)


def _closed_around(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, level: float, rax: float, zax: float
) -> np.ndarray | None:
    if not np.isfinite(level):
        return None
    try:
        segs = contour_generator(x=r, y=z, z=psi).lines(float(level))
    except Exception:
        return None
    pad_r = 0.51 * abs(float(r[1] - r[0])) if r.size > 1 else 0.02
    pad_z = 0.51 * abs(float(z[1] - z[0])) if z.size > 1 else 0.02
    best = None
    for seg in segs:
        arr = np.asarray(seg, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 12:
            continue
        if np.hypot(arr[0, 0] - arr[-1, 0], arr[0, 1] - arr[-1, 1]) > 2.0 * pad_r:
            continue
        if (
            arr[:, 0].min() <= r[0] + pad_r
            or arr[:, 0].max() >= r[-1] - pad_r
            or arr[:, 1].min() <= z[0] + pad_z
            or arr[:, 1].max() >= z[-1] - pad_z
        ):
            continue
        if not _contains(arr, rax, zax):
            continue
        if best is None or arr.shape[0] > best.shape[0]:
            best = arr
    return best


def _grad_mag(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> np.ndarray:
    d_z, d_r = np.gradient(psi, z, r)
    return np.hypot(d_r, d_z)


def _bilinear(r: np.ndarray, z: np.ndarray, field: np.ndarray, rq: np.ndarray, zq: np.ndarray) -> np.ndarray:
    ir = np.clip(np.searchsorted(r, rq) - 1, 0, r.size - 2)
    iz = np.clip(np.searchsorted(z, zq) - 1, 0, z.size - 2)
    tr = (rq - r[ir]) / (r[ir + 1] - r[ir] + 1.0e-30)
    tz = (zq - z[iz]) / (z[iz + 1] - z[iz] + 1.0e-30)
    v00 = field[iz, ir]
    v10 = field[iz, ir + 1]
    v01 = field[iz + 1, ir]
    v11 = field[iz + 1, ir + 1]
    return (1.0 - tz) * ((1.0 - tr) * v00 + tr * v10) + tz * ((1.0 - tr) * v01 + tr * v11)


def w_on_track(r: np.ndarray, z: np.ndarray, psi: np.ndarray, poly: np.ndarray) -> float:
    """W = ∮ dℓ / (R |∇ψ|) on the given polyline."""
    g = _grad_mag(r, z, psi)
    mid = 0.5 * (poly[:-1] + poly[1:])
    dl = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    rq, zq = mid[:, 0], mid[:, 1]
    grad = np.maximum(_bilinear(r, z, g, rq, zq), 1.0e-12)
    return float(np.sum(dl / (np.maximum(rq, 1.0e-6) * grad)))


def p_prime_of_tau(tau: np.ndarray) -> np.ndarray:
    d = np.asarray(D_MODEL, dtype=np.float64)
    poly = np.ones_like(tau)
    for k, ck in enumerate(d, start=1):
        poly = poly + ck * tau**k
    return (1.0 - tau) * poly


def s_of_tau(tau: np.ndarray) -> np.ndarray:
    """s(τ)=∫_τ^1 P'(σ) dσ with d=(0,0) ⇒ P'=1-σ ⇒ s=(1-τ)^2/2."""
    return 0.5 * (1.0 - tau) ** 2


def first_q95(a: float, b_t: float, ip_ma: float) -> float:
    ip = max(abs(float(ip_ma)), 1.0e-3)
    q_star = 5.0 * (a**2) * abs(b_t) / (R0_M * ip)
    return float(np.clip(q_star, *Q95_CLIP))


def thomson_pressure(row: dict, t: float) -> float:
    """⟨p⟩ ≈ 2 n_e T_e e from finite midplane chords at this efit time."""
    pressures = []
    for prefix, r_key in (("thomson_core", "thomson_core_R"), ("thomson_edge", "thomson_edge_spatial")):
        t_key = f"{prefix}_times"
        if t_key not in row or f"{prefix}_Te" not in row:
            continue
        times = _as1(row[t_key])
        te = np.asarray(row[f"{prefix}_Te"], dtype=np.float64)
        ne = np.asarray(row[f"{prefix}_ne"], dtype=np.float64)
        if times.size == 0 or te.size == 0:
            continue
        te_t = interpolate(times, te, np.array([t]))[0]
        ne_t = interpolate(times, ne, np.array([t]))[0]
        rch = np.asarray(row.get(r_key, []), dtype=np.float64).reshape(-1)
        n = min(te_t.size, ne_t.size)
        for j in range(n):
            if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
                continue
            if rch.size == n and not (0.25 <= float(rch[j]) <= 1.45):
                continue
            pressures.append(2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE)
    if not pressures:
        return float("nan")
    return float(np.median(pressures))


def _beta_n_raw(p_mean: float, b_t: float, ip_ma: float, a: float) -> float:
    if not np.isfinite(p_mean) or p_mean <= 0 or abs(b_t) < 1.0e-6 or abs(ip_ma) < 1.0e-3 or a < 1.0e-3:
        return float("nan")
    return float(100.0 * p_mean * 2.0 * MU0 * a / (abs(b_t) * abs(ip_ma)))


def first_beta_n(p_mean: float, b_t: float, ip_ma: float, a: float) -> float:
    raw = _beta_n_raw(p_mean, b_t, ip_ma, a)
    if not np.isfinite(raw):
        return BETAN_TYPICAL
    return float(np.clip(raw, *BETAN_CLIP))


def frame_currents_a(row: dict, t_index: int, t_dst: np.ndarray) -> tuple[dict[str, float], float, float]:
    currents = {}
    for column in (*P_COIL_COLUMNS, SOL_COLUMN):
        y = _interp_col(row, column, t_dst)
        currents[column] = float(y[t_index]) * 1.0e3  # kA or kA·turn → A or A·turn
    tf_a = float(_interp_col(row, TF_COLUMN, t_dst)[t_index]) * 1.0e3
    ip_a = float(_interp_col(row, IP_COLUMN, t_dst)[t_index]) * 1.0e3
    return currents, tf_a, ip_a


def knobs_for_frame(row: dict, t: float, b_t: float, ip_ma: float) -> tuple[float, float]:
    p_mean = thomson_pressure(row, t)
    q95 = first_q95(A_TRIAL_M, b_t, ip_ma)
    beta_n = first_beta_n(p_mean, b_t, ip_ma, A_TRIAL_M)
    return q95, beta_n


