"""E6: curve-distance training of the M9 network on FAIR-MAST only.

Loss (per minibatch of whole shots, 32 shots per step, at most 128 consecutive frames per shot):
  PCA coefficient MSE (anchor)
  + w_cd * L_cd   range-normalised per-shot curve distance and its frame-to-frame derivative on
                  q95, betaN (scalar heads) and kappa, delta_top, delta_bot, V, R_axis, Z_axis read
                  off the predicted psi with differentiable (soft) extractors
  + w_lcfs * (1 - soft IoU of the LCFS interior)
  + w_b * boundary band, w_tip * boundary band restricted to the top and bottom 15 % of the EFIT
  LCFS height (where the scorer reads delta_top / delta_bot), w_c * O/X critical points, w_t * temporal coefficient differences,
  + w_s * Huber on the five standardised scalar heads (as E4).
Late-shot weighting: frame weight 1 + alpha_j * clip((t_frac - 0.8) / 0.2, 0, 1) with alpha = 4 for
betaN and 2 for the other series; series weights omega = 2 for betaN, delta_top, delta_bot, else 1.
Schedule: AdamW 1e-3, ReduceLROnPlateau on the validation objective J (factor 0.5, patience 4,
min 2e-5), EMA of the weights (decay 0.998 per step); validation on the EMA weights every epoch.
Early stopping: stop when J (archval100, FAIR-MAST) has not improved by more than 0.2 % relative for
PATIENCE = 20 epochs (at least 30 epochs, at most 300). The three best EMA states by J are kept.
Every epoch writes last.pt, so a run resumes after a crash.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import COMMON, EXP, REPO, SPLITS_JSON, assert_safe
from experiments.m9x import A_MINOR, DR, DZ, SCAL, _sample, load_scalars
from models.m9.network import _clip_features, augment_temporal, build_dualctx

E6 = EXP / "E6"
SERIES = ("q95", "beta_n", "kappa", "tri_top", "tri_bot", "volume", "R_axis", "Z_axis")
SOFT = SERIES[2:]
OMEGA = np.array([1, 2, 1, 2, 2, 1, 1, 1], np.float32)
ALPHA = np.array([2, 4, 2, 2, 2, 2, 2, 2], np.float32)
OMEGA_LCFS, ALPHA_LCFS = 1.0, 2.0
LATE = 0.8
LAM_D = 0.5
TAU_S, T_EXT, T_AX = 0.02, 0.03, 0.02
TIP_FRAC = 0.15  # tip band: top / bottom 15 % of the EFIT LCFS height
XPT_TRI = False  # True: R of the highest / lowest point = refined EFIT X-point (does not track the scorer)
DIL = 2  # region = EFIT LCFS interior dilated by DIL cells
SHOTS_PER_STEP, MAX_FRAMES = 32, 128
PATIENCE, TOL, MIN_EPOCHS, MAX_EPOCHS = 20, 2e-3, 30, 300
EMA_DECAY = 0.998
RANGE_FLOOR = 0.25  # per-shot range floor, in units of the series' global std


def log_default(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------ grid
def grid(dev):
    z = np.load(REPO / "fec_scoring" / "masks" / "mast_envelope.npz")
    R = np.asarray(z["grid_R"], np.float32).reshape(-1)
    Zg = np.asarray(z["grid_Z"], np.float32).reshape(-1)
    RR = torch.from_numpy(np.tile(R, len(Zg))).to(dev)          # psi[iz, ir] flattened row-major
    ZZ = torch.from_numpy(np.repeat(Zg, len(R))).to(dev)
    dA = float((R[1] - R[0]) * (Zg[1] - Zg[0]))
    env = torch.from_numpy(np.asarray(z["mask_coarse"]).astype(bool).reshape(-1)).to(dev)
    return RR, ZZ, dA, env


_U, _V = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))  # 5x5 patch, u along R, v along Z
_M = np.column_stack([np.ones(25), _U.ravel(), _V.ravel(), _U.ravel() ** 2, (_U * _V).ravel(), _V.ravel() ** 2])
_P = np.linalg.pinv(_M).astype(np.float32)  # (6, 25)


def saddle_refine(psi3, P0, G):
    """Stationary point of a least-squares quadratic fitted to the 5x5 patch of psi3 (B, 65, 65)
    around the grid node nearest P0 (B, 2) [R, Z]. Returns (B, 2) refined [R, Z] and a (B,) valid
    mask (finite P0). Differentiable in psi3; displacement limited to 2 cells (tanh)."""
    RR, ZZ, _, _ = G
    n = 65
    Rv, Zv = RR[:n], ZZ[::n]
    dR, dZ = Rv[1] - Rv[0], Zv[1] - Zv[0]
    ok = torch.isfinite(P0).all(1)
    P = torch.where(ok[:, None], P0, torch.zeros_like(P0))
    ir = ((P[:, 0] - Rv[0]) / dR).round().long().clamp(2, n - 3)
    iz = ((P[:, 1] - Zv[0]) / dZ).round().long().clamp(2, n - 3)
    du = torch.arange(-2, 3, device=psi3.device)
    IZ = (iz[:, None, None] + du[None, :, None]).expand(-1, 5, 5)
    IR = (ir[:, None, None] + du[None, None, :]).expand(-1, 5, 5)
    b = torch.arange(len(P), device=psi3.device)[:, None, None].expand(-1, 5, 5)
    patch = psi3[b, IZ, IR].reshape(len(P), 25)
    patch = patch - patch.mean(1, keepdim=True)
    c = patch @ torch.from_numpy(_P).to(psi3.device).T
    c1, c2, c3, c4, c5 = c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
    det = 4 * c3 * c5 - c4 ** 2
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, -1e-12), det)
    su = -(2 * c5 * c1 - c4 * c2) / det
    sv = -(2 * c3 * c2 - c4 * c1) / det
    su, sv = 2 * torch.tanh(su / 2), 2 * torch.tanh(sv / 2)
    return torch.stack([Rv[ir] + su * dR, Zv[iz] + sv * dZ], 1), ok


def soft_feats(psi, pax, pb, region, G, xu=None, xl=None):
    """Differentiable shape read-outs of psi (B, 4225) normalised with the EFIT psi_ax, psi_b.

    Returns (B, 6) [kappa, tri_top, tri_bot, V, R_axis, Z_axis] and the soft interior s (B, 4225).
    Interior s = sigmoid((1 - psi_N) / tau) restricted to `region` (the EFIT LCFS interior dilated
    by DIL cells). Each grid column gives a sub-grid top and bottom of the interior (centroid +/- half
    the soft height) and each grid row a sub-grid inner and outer edge; Z_max, Z_min, R_max, R_min
    and the R of the highest and lowest points are soft-argmax read-outs over those columns / rows
    (temperature T_EXT). The axis is the soft-argmin of psi_N. V = 2 pi sum R s dA."""
    RR, ZZ, dA, _ = G
    n = 65
    dR = float(RR[1] - RR[0])
    dZ = float(ZZ[n] - ZZ[0])
    dps = (pax - pb).clamp_min(1e-6)
    psin = (pax[:, None] - psi) / dps[:, None]
    x = (1 - psin) / TAU_S
    s = torch.sigmoid(x) * region
    V = 2 * math.pi * (s * RR).sum(1) * dA
    S = s.view(-1, n, n)                       # [b, iz, ir]
    Rv, Zv = RR[:n], ZZ[::n]
    h = S.sum(1) * dZ                          # column heights (b, ir)
    zc = (S * Zv[None, :, None]).sum(1) / (S.sum(1) + 1e-6)
    ztop, zbot = zc + h / 2, zc - h / 2
    lc = torch.log(h / (h + 0.05) + 1e-9)       # down-weights empty columns
    wt = torch.softmax(ztop / T_EXT + lc, 1)
    wb = torch.softmax(-zbot / T_EXT + lc, 1)
    Zmax, Zmin = (wt * ztop).sum(1), (wb * zbot).sum(1)
    RaZmax, RaZmin = (wt * Rv).sum(1), (wb * Rv).sum(1)
    wdt = S.sum(2) * dR                        # row widths (b, iz)
    rc = (S * Rv[None, None, :]).sum(2) / (S.sum(2) + 1e-6)
    rout, rin = rc + wdt / 2, rc - wdt / 2
    lr = torch.log(wdt / (wdt + 0.05) + 1e-9)
    Rmax = (torch.softmax(rout / T_EXT + lr, 1) * rout).sum(1)
    Rmin = (torch.softmax(-rin / T_EXT + lr, 1) * rin).sum(1)
    if XPT_TRI and xu is not None:
        psi3 = psi.view(-1, n, n)
        pu, oku = saddle_refine(psi3, xu, G)
        pl, okl = saddle_refine(psi3, xl, G)
        RaZmax = torch.where(oku, pu[:, 0], RaZmax)
        RaZmin = torch.where(okl, pl[:, 0], RaZmin)
    a = ((Rmax - Rmin) / 2).clamp_min(1e-3)
    Rg = (Rmax + Rmin) / 2
    kappa = (Zmax - Zmin) / (2 * a)
    tt, tb = (Rg - RaZmax) / a, (Rg - RaZmin) / a
    logs = F.logsigmoid(x).masked_fill(region <= 0, -1e4)
    wax = torch.softmax(-psin / T_AX + logs, 1)
    Rax, Zax = (wax * RR).sum(1), (wax * ZZ).sum(1)
    return torch.stack([kappa, tt, tb, V, Rax, Zax], 1), s


def soft_iou_dist(s, tin):
    inter = (s * tin).sum(1)
    union = (s + tin - s * tin).sum(1).clamp_min(1.0)
    return 1 - inter / union


# ------------------------------------------------------------------------------------------ data
class Data:
    """Everything the trainer and the evaluator need, on CPU (numpy) unless noted."""

    def __init__(self, dev, need_train_features=True, log=log_default):
        t0 = time.time()
        self.dev = dev
        d = np.load(COMMON / "data.npz")
        X, self.Z, self.ids, self.fi = d["X"], d["Z"], d["ids"], d["fi"]
        self.keys = [str(k) for k in d["keys"]]
        self.mu = torch.from_numpy(d["pca_mean"]).to(dev)
        self.C = torch.from_numpy(d["pca_components"]).to(dev)
        self.pca_mean, self.pca_components = d["pca_mean"], d["pca_components"]
        self.K, self.n_raw = self.C.shape[0], X.shape[1]
        sp = json.loads(SPLITS_JSON.read_text())
        archval, av100 = set(sp["archval"]), set(sp["archval100"])
        val_shot = np.array([k in archval for k in self.keys])
        v100_shot = np.array([k in av100 for k in self.keys])
        self.val_m = val_shot[self.ids]
        self.v100 = np.flatnonzero(v100_shot[self.ids])
        n = len(self.ids)
        self.starts = np.flatnonzero(np.r_[True, self.ids[1:] != self.ids[:-1]])
        self.lens = np.diff(np.r_[self.starts, n])
        assert np.all(self.ids[self.starts] == np.arange(len(self.keys)))
        pos = np.arange(n) - self.starts[self.ids]
        self.tfrac = (pos / np.maximum(self.lens[self.ids] - 1, 1)).astype(np.float32)
        self.train_shots = np.flatnonzero(~val_shot)
        # features
        X_aug = augment_temporal(X, self.ids, 9, "smoothdiff")
        del X
        self.X_v100_raw = np.asarray(X_aug[self.v100], np.float32)
        X64 = np.asarray(X_aug[~self.val_m], np.float64)
        self.mean = np.nan_to_num(X64.mean(0)).astype(np.float32)
        std = X64.std(0)
        self.std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std).astype(np.float32)
        del X64
        if need_train_features:
            self.xs = torch.from_numpy(np.ascontiguousarray(
                _clip_features(((X_aug - self.mean) / self.std).astype(np.float32), 10.0)))
        self.n_feat = X_aug.shape[1]
        del X_aug
        # route targets
        r = np.load(COMMON / "route.npz")
        self.r_ok = r["ok"]
        self.psi_ax = np.where(self.r_ok, r["psi_ax"], 1.0).astype(np.float32)
        self.psi_b = np.where(self.r_ok, r["psi_b"], 0.0).astype(np.float32)
        self.inside_packed = r["inside"]
        self.opt, self.xu, self.xl = (r[k].astype(np.float32) for k in ("opt", "xu", "xl"))
        self.G = grid(dev)
        # scalar labels (q95, beta_n, R_axis, Z_axis, li) and soft truth
        self.S = load_scalars(self.keys, self.ids, self.fi)
        tr = self.S[~self.val_m]
        self.s_mean = np.nanmean(tr, 0).astype(np.float32)
        self.s_std = np.maximum(np.nanstd(tr, 0), 1e-6).astype(np.float32)
        self.truth_soft = self._truth_soft(log)
        T = np.concatenate([self.S[:, :2], self.truth_soft], 1)
        Vm = np.isfinite(T)
        Vm[:, 2:] &= self.r_ok[:, None]
        self.T, self.Tvalid = np.where(Vm, T, 0).astype(np.float32), Vm
        gstd = np.array([np.nanstd(np.where(Vm[:, j] & ~self.val_m, T[:, j], np.nan)) for j in range(T.shape[1])])
        rng = np.zeros((len(self.keys), T.shape[1]), np.float32)
        for s, (a, L) in enumerate(zip(self.starts, self.lens)):
            seg, ok = T[a:a + L], Vm[a:a + L]
            for j in range(T.shape[1]):
                v = seg[ok[:, j], j]
                rng[s, j] = (v.max() - v.min()) if len(v) else 0.0
        self.shot_range = np.maximum(rng, RANGE_FLOOR * gstd[None, :]).astype(np.float32)
        self.gstd = gstd
        log(f"[e6] data ready: {n} frames, {len(self.keys)} shots, {len(self.train_shots)} train shots, "
            f"{len(self.v100)} archval100 frames, {round(time.time() - t0)} s")

    def region(self, idx):
        tin = np.unpackbits(self.inside_packed[idx], axis=1)[:, :65 * 65].astype(np.float32)
        t = torch.from_numpy(tin).to(self.dev)
        reg = F.max_pool2d(t.view(-1, 1, 65, 65), 2 * DIL + 1, 1, DIL).view(-1, 65 * 65)
        ok = torch.from_numpy(self.r_ok[idx]).to(self.dev)
        env = self.G[3].float()[None, :].expand_as(reg)
        reg = torch.where(ok[:, None] & (reg.sum(1, keepdim=True) > 0), reg, env)
        return t, reg

    def xpts(self, idx):
        return (torch.from_numpy(self.xu[idx]).to(self.dev), torch.from_numpy(self.xl[idx]).to(self.dev))

    def pab(self, idx):
        return (torch.from_numpy(self.psi_ax[idx]).to(self.dev), torch.from_numpy(self.psi_b[idx]).to(self.dev))

    def _truth_soft(self, log):
        f = E6 / "truth_soft.npz"
        if f.is_file():
            z = np.load(f)
            if len(z["feats"]) == len(self.ids):
                return z["feats"]
        log("[e6] computing soft truth read-outs for every frame ...")
        out = np.zeros((len(self.ids), len(SOFT)), np.float32)
        Zt = torch.from_numpy(self.Z)
        with torch.no_grad():
            for a in range(0, len(self.ids), 4096):
                idx = np.arange(a, min(a + 4096, len(self.ids)))
                psi = Zt[idx].to(self.dev) @ self.C + self.mu
                _, reg = self.region(idx)
                pax, pb = self.pab(idx)
                out[idx] = soft_feats(psi, pax, pb, reg, self.G, *self.xpts(idx))[0].cpu().numpy()
        assert_safe(E6).mkdir(parents=True, exist_ok=True)
        np.savez(f, feats=out)
        return out


# ------------------------------------------------------------------------------- curve distance
def late_w(tfrac, alpha):
    ramp = np.clip((tfrac - LATE) / (1 - LATE), 0, 1)
    return 1 + ramp[:, None] * np.asarray(alpha, np.float32)[None, :]


def per_shot_curve(e, valid, w, local, n_loc):
    """Per-shot weighted mean of e^2 and of (frame-to-frame difference of e)^2.

    e, valid, w: (B, J); local: (B,) shot index in 0..n_loc-1, frames of a shot contiguous and in
    time order. Returns (n_loc, J) value, derivative and a (n_loc, J) has-data mask."""
    vw = w * valid
    e = torch.where(valid > 0, e, torch.zeros_like(e))
    J = e.shape[1]
    num = torch.zeros(n_loc, J, device=e.device).index_add_(0, local, vw * e ** 2)
    den = torch.zeros(n_loc, J, device=e.device).index_add_(0, local, vw)
    same = (local[1:] == local[:-1]).float()[:, None] * valid[1:] * valid[:-1]
    dw = same * w[1:]
    dnum = torch.zeros(n_loc, J, device=e.device).index_add_(0, local[1:], dw * (e[1:] - e[:-1]) ** 2)
    dden = torch.zeros(n_loc, J, device=e.device).index_add_(0, local[1:], dw)
    return num / den.clamp_min(1e-9), dnum / dden.clamp_min(1e-9), den > 0, dden > 0


def curve_loss(pred, D: Data, idx, local, n_loc, shots):
    """Training form: squared, range-normalised, late-weighted, omega-weighted mean."""
    dev = pred.device
    T = torch.from_numpy(D.T[idx]).to(dev)
    valid = torch.from_numpy(D.Tvalid[idx].astype(np.float32)).to(dev)
    w = torch.from_numpy(late_w(D.tfrac[idx], ALPHA)).to(dev)
    rng = torch.from_numpy(D.shot_range[shots]).to(dev)[local]
    e = (pred - T) / rng
    v, dv, hv, hd = per_shot_curve(e, valid, w, local, n_loc)
    om = torch.from_numpy(OMEGA).to(dev)
    vj = (v * hv).sum(0) / hv.sum(0).clamp_min(1)
    dj = (dv * hd).sum(0) / hd.sum(0).clamp_min(1)
    return ((vj + LAM_D * dj) * om).sum() / om.sum(), vj.detach(), dj.detach()


@torch.no_grad()
def objective(z, heads2, D: Data, idx=None) -> dict:
    """Validation objective J on archval100 frames (default) for predicted coefficients z (n, K)
    and de-standardised q95, betaN (n, 2). J = omega-weighted mean over the 8 series of the
    per-shot mean of [nRMSE + 0.5 nRMSE of the frame-to-frame difference] (late-weighted, range
    normalised), plus the LCFS term (per-shot late-weighted mean of 1 - soft IoU), omega_LCFS = 1."""
    dev = D.dev
    idx = D.v100 if idx is None else idx
    soft = torch.zeros(len(idx), len(SOFT), device=dev)
    lc = torch.zeros(len(idx), device=dev)
    for a in range(0, len(idx), 2048):
        ii = idx[a:a + 2048]
        psi = z[a:a + 2048].to(dev) @ D.C + D.mu
        tin, reg = D.region(ii)
        pax, pb = D.pab(ii)
        f, s = soft_feats(psi, pax, pb, reg, D.G, *D.xpts(ii))
        soft[a:a + len(ii)] = f
        lc[a:a + len(ii)] = soft_iou_dist(s, tin)
    pred = torch.cat([heads2.to(dev), soft], 1)
    shots_all = D.ids[idx]
    shots, local = np.unique(shots_all, return_inverse=True)
    local_t = torch.from_numpy(local).to(dev)
    T = torch.from_numpy(D.T[idx]).to(dev)
    valid = torch.from_numpy(D.Tvalid[idx].astype(np.float32)).to(dev)
    w = torch.from_numpy(late_w(D.tfrac[idx], ALPHA)).to(dev)
    rng = torch.from_numpy(D.shot_range[shots]).to(dev)[local_t]
    v, dv, hv, hd = per_shot_curve((pred - T) / rng, valid, w, local_t, len(shots))
    nr = (v.sqrt() * hv).sum(0) / hv.sum(0).clamp_min(1)
    nd = (dv.sqrt() * hd).sum(0) / hd.sum(0).clamp_min(1)
    okf = torch.from_numpy(D.r_ok[idx].astype(np.float32)).to(dev)
    wl = torch.from_numpy(late_w(D.tfrac[idx], [ALPHA_LCFS])[:, 0]).to(dev) * okf
    ln = torch.zeros(len(shots), device=dev).index_add_(0, local_t, wl * lc)
    ld = torch.zeros(len(shots), device=dev).index_add_(0, local_t, wl)
    lcfs = float((ln[ld > 0] / ld[ld > 0]).mean())
    per = (nr + LAM_D * nd).cpu().numpy()
    J = (float((per * OMEGA).sum()) + OMEGA_LCFS * lcfs) / (OMEGA.sum() + OMEGA_LCFS)
    zt = torch.from_numpy(D.Z[idx]).to(dev)
    mse = float(((z.to(dev) - zt) ** 2).mean())
    return {"J": J, "lcfs": lcfs, "mse": mse, **{f"nrmse_{s}": float(x) for s, x in zip(SERIES, nr.cpu().numpy())},
            **{f"nrmse_d_{s}": float(x) for s, x in zip(SERIES, nd.cpu().numpy())}}


# --------------------------------------------------------------------------------- member eval
@torch.no_grad()
def member_outputs(ck, D: Data, model=None):
    """(z, heads2) on archval100 frames for a member checkpoint dict (or a live model + D norms)."""
    dev = D.dev
    if model is None:
        model = build_dualctx(int(ck["n_features"]), int(ck["n_pca"]), int(ck["n_raw_features"]))
        model.load_state_dict(ck["model_state"])
        model = model.to(dev)
        fm, fs, clip = ck["feature_mean"], ck["feature_std"], float(ck["feature_clip"])
        sm, ss = ck.get("scalar_mean"), ck.get("scalar_std")
    else:
        fm, fs, clip, sm, ss = D.mean, D.std, 10.0, D.s_mean, D.s_std
    model.eval()
    zs, hs = [], []
    for a in range(0, len(D.v100), 4096):
        x = _clip_features(((D.X_v100_raw[a:a + 4096] - fm) / fs).astype(np.float32), clip)
        o = model(torch.from_numpy(np.ascontiguousarray(x)).to(dev))
        zs.append(o[0].float())
        if sm is not None:
            hs.append(torch.stack(o[1:3], 1).float() * torch.from_numpy(ss[:2]).to(dev)
                      + torch.from_numpy(sm[:2]).to(dev))
        else:
            hs.append(torch.tensor([[3.5, 1.5]], device=dev).expand(len(x), 2))
    return torch.cat(zs), torch.cat(hs)


def ensemble_objective(files, D: Data) -> dict:
    zs, hs = [], []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        z, h = member_outputs(ck, D)
        zs.append(z); hs.append(h)
    return objective(torch.stack(zs).mean(0), torch.stack(hs).mean(0), D)


# ------------------------------------------------------------------------------------- trainer
def train(out_dir: Path, seed: int, w: dict, D: Data, max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS,
          patience=PATIENCE, log=log_default) -> dict:
    out_dir = assert_safe(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "meta.pt").is_file():
        return torch.load(out_dir / "meta.pt", map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    dev = D.dev
    zt = torch.from_numpy(D.Z).to(dev)
    st = torch.from_numpy(np.nan_to_num((D.S - D.s_mean) / D.s_std, nan=np.nan)).to(dev)
    torch.manual_seed(int(seed))
    model = build_dualctx(D.n_feat, D.K, D.n_raw).to(dev)
    evalm = build_dualctx(D.n_feat, D.K, D.n_raw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4,
                                                       threshold=1e-3, min_lr=2e-5)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(int(seed) + 7)
    state = {"ep": 0, "hist": [], "best_J": 1e30, "best_J_ep": -1, "bad": 0, "top": [], "best_mse": 1e30,
             "best_mse_ep": -1, "stopped": None}
    last = out_dir / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ema = {k: v.to(dev) for k, v in ck["ema"].items()}
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
        log(f"[e6] resumed {out_dir.name} at epoch {state['ep']}")
    use_c = w.get("w_c", 0) > 0
    for ep in range(state["ep"], max_epochs):
        if state["stopped"] is not None:
            break
        model.train()
        perm = rng.permutation(D.train_shots)
        acc = {k: 0.0 for k in ("mse", "cd", "lcfs", "b", "tip", "c", "t", "s")}
        n_steps = int(math.ceil(len(perm) / SHOTS_PER_STEP))
        for b in range(n_steps):
            shots = np.sort(perm[b * SHOTS_PER_STEP:(b + 1) * SHOTS_PER_STEP])
            parts, loc = [], []
            for i, s in enumerate(shots):
                a, L = D.starts[s], D.lens[s]
                if L > MAX_FRAMES:
                    a = a + int(rng.integers(0, L - MAX_FRAMES + 1)); L = MAX_FRAMES
                parts.append(np.arange(a, a + L)); loc.append(np.full(L, i))
            idx = np.concatenate(parts)
            local = torch.from_numpy(np.concatenate(loc)).to(dev)
            outs = model(D.xs[torch.from_numpy(idx)].to(dev, non_blocking=True))
            zh = outs[0]
            ig = torch.from_numpy(idx).to(dev)
            zi = zt[ig]
            loss = ((zh - zi) ** 2).mean()
            acc["mse"] += float(loss.detach())
            psi = zh @ D.C + D.mu
            tin, reg = D.region(idx)
            pax, pb = D.pab(idx)
            soft, s_in = soft_feats(psi, pax, pb, reg, D.G, *D.xpts(idx))
            heads = torch.stack(outs[1:6], 1)
            h2 = heads[:, :2] * torch.from_numpy(D.s_std[:2]).to(dev) + torch.from_numpy(D.s_mean[:2]).to(dev)
            pred = torch.cat([h2, soft], 1)
            lcd, _, _ = curve_loss(pred, D, idx, local, len(shots), shots)
            loss = loss + w["w_cd"] * lcd; acc["cd"] += float(lcd)
            okf = torch.from_numpy(D.r_ok[idx]).to(dev)
            if okf.any():
                ll = soft_iou_dist(s_in[okf], tin[okf]).mean()
                loss = loss + w["w_lcfs"] * ll; acc["lcfs"] += float(ll)
                pt = zi[okf] @ D.C + D.mu
                ph = psi[okf]
                pa, pbb = pax[okf][:, None], pb[okf][:, None]
                dps = (pa - pbb).clamp_min(1e-6)
                if w.get("w_b", 0) > 0:
                    band = ((((pa - pt) / dps) - 1).abs() < 0.1) & D.G[3][None, :]
                    lb = (band * (ph - pt) ** 2).sum(1).mean() / D.K
                    loss = loss + w["w_b"] * lb; acc["b"] += float(lb)
                if w.get("w_tip", 0) > 0:
                    tin_o = tin[okf] > 0.5
                    Zg = D.G[1][None, :].expand_as(tin_o)
                    zmx = torch.where(tin_o, Zg, torch.full_like(Zg, -9)).amax(1, keepdim=True)
                    zmn = torch.where(tin_o, Zg, torch.full_like(Zg, 9)).amin(1, keepdim=True)
                    hgt = (zmx - zmn).clamp_min(0.1)
                    tipz = (Zg > zmx - TIP_FRAC * hgt) | (Zg < zmn + TIP_FRAC * hgt)
                    band2 = ((((pa - pt) / dps) - 1).abs() < 0.1) & D.G[3][None, :] & tipz
                    ltp = (band2 * (ph - pt) ** 2).sum(1).mean() / D.K
                    loss = loss + w["w_tip"] * ltp; acc["tip"] += float(ltp)
                if use_c:
                    io = idx[okf.cpu().numpy()]
                    g = ph.view(-1, 65, 65)
                    gz, gr = torch.gradient(g, spacing=(DZ, DR), dim=(1, 2))
                    gm = (gz ** 2 + gr ** 2) * (A_MINOR / dps.view(-1, 1, 1)) ** 2
                    terms = []
                    for arr in (D.opt, D.xu, D.xl):
                        P = torch.from_numpy(arr[io]).to(dev)
                        fin = torch.isfinite(P).all(1)
                        if fin.any():
                            terms.append(_sample(gm[fin], P[fin, 0], P[fin, 1]))
                    for arr in (D.xu, D.xl):
                        P = torch.from_numpy(arr[io]).to(dev)
                        fin = torch.isfinite(P).all(1)
                        if fin.any():
                            v = _sample(g[fin], P[fin, 0], P[fin, 1])
                            terms.append(((v - pbb.view(-1)[fin]) / dps.view(-1)[fin]) ** 2)
                    if terms:
                        lc = torch.cat(terms).mean()
                        loss = loss + w["w_c"] * lc; acc["c"] += float(lc)
            same = local[1:] == local[:-1]
            if w.get("w_t", 0) > 0 and same.any():
                lt = (((zh[1:] - zh[:-1]) - (zi[1:] - zi[:-1]))[same] ** 2).mean()
                loss = loss + w["w_t"] * lt; acc["t"] += float(lt)
            if w.get("w_s", 0) > 0:
                tgt = st[ig]
                fin = torch.isfinite(tgt)
                if fin.any():
                    ls = F.huber_loss(heads[fin], tgt[fin], delta=1.0)
                    loss = loss + w["w_s"] * ls; acc["s"] += float(ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(EMA_DECAY).add_(v.detach().float(), alpha=1 - EMA_DECAY)
                    else:
                        ema[k].copy_(v)
        evalm.load_state_dict(ema)
        z, h = member_outputs(None, D, model=evalm)
        ob = objective(z, h, D)
        J = ob["J"]
        lr = opt.param_groups[0]["lr"]
        sched.step(J)
        rec = {"ep": ep, "lr": lr, **ob, **{f"tr_{k}": v / n_steps for k, v in acc.items()}}
        state["hist"].append(rec)
        ema_cpu = {k: v.detach().cpu().clone() for k, v in ema.items()}
        # top-3 by raw J
        top = state["top"]
        if len(top) < 3 or J < max(t["J"] for t in top):
            f = out_dir / f"ema_ep{ep:03d}.pt"
            torch.save(ema_cpu, f)
            top.append({"ep": ep, "J": J, "file": f.name})
            top.sort(key=lambda t: t["J"])
            for t in top[3:]:
                (out_dir / t["file"]).unlink(missing_ok=True)
            state["top"] = top[:3]
        if ob["mse"] < state["best_mse"]:
            state["best_mse"], state["best_mse_ep"] = ob["mse"], ep
            torch.save(ema_cpu, out_dir / "ema_best_mse.pt")
        if J < state["best_J"] * (1 - TOL):
            state["best_J"], state["best_J_ep"], state["bad"] = J, ep, 0
        else:
            state["bad"] += 1
        state["ep"] = ep + 1
        if state["bad"] >= patience and ep + 1 >= min_epochs:
            state["stopped"] = ep
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema_cpu, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "state": state}, out_dir / "last.part.pt")
        (out_dir / "last.part.pt").replace(last)
        log(f"[e6] {out_dir.name} ep {ep:3d} lr {lr:.1e} J {J:.5f} (best {state['best_J']:.5f}@{state['best_J_ep']}, "
            f"bad {state['bad']}) mse {ob['mse']:.3e} bN {ob['nrmse_beta_n']:.4f} q95 {ob['nrmse_q95']:.4f} "
            f"dt {ob['nrmse_tri_top']:.4f} db {ob['nrmse_tri_bot']:.4f} lcfs {ob['lcfs']:.4f} | "
            + " ".join(f"{k}={v / n_steps:.3g}" for k, v in acc.items() if v))
        if state["stopped"] is not None:
            log(f"[e6] {out_dir.name} early stop at epoch {ep} (no >{TOL:.1%} gain in J for {patience} epochs)")
            break
    if state["stopped"] is None:
        state["stopped"] = state["ep"] - 1
    meta = {"feature_mean": D.mean, "feature_std": D.std, "n_raw_features": int(D.n_raw),
            "n_features": int(D.n_feat), "n_pca": int(D.K), "pca_mean": D.pca_mean,
            "pca_components": D.pca_components, "arch": "dualctx", "temporal_window": 9,
            "temporal_mode": "smoothdiff", "feature_clip": 10.0, "init_seed": int(seed), "weights": w,
            "scalar_mean": D.s_mean, "scalar_std": D.s_std, "history": state["hist"], "top": state["top"],
            "best_mse_epoch": state["best_mse_ep"], "stopped_epoch": state["stopped"],
            "best_J": state["best_J"], "best_J_epoch": state["best_J_ep"],
            "train_seconds": round(time.perf_counter() - t0, 1)}
    torch.save(meta, out_dir / "meta.pt")
    return meta


def member_file(mdir: Path, state_file: str, label: str) -> Path:
    f = mdir / f"cand_{label}.member.pt"
    if not f.is_file():
        ck = dict(torch.load(mdir / "meta.pt", map_location="cpu", weights_only=False))
        ck.pop("history", None)
        ck["model_state"] = torch.load(mdir / state_file, map_location="cpu", weights_only=False)
        ck["selected"] = label
        torch.save(ck, f)
    return f
