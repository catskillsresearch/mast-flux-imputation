"""M9 trainer with optional route terms (arm B) and a predictor that also reads the scalar heads.

Arm A (all weights 0) is the shipped recipe (dualctx 512x4, window 9 smoothdiff, AdamW 1e-3/1e-4,
clip 1.0, batch 2048 shot-balanced, 50 epochs) on the common data of ``experiments.prep``.
Arm B adds, on psi_hat = mu + z_hat C (and the EFIT psi_ax, psi_b, LCFS interior, O/X-points of
``experiments.route_labels``):
  w_b  boundary band: squared error on pixels with |psi_N - 1| < 0.1 (coefficient-loss units)
  w_a  soft LCFS: 1 - softIoU(sigmoid((1 - psi_hat_N)/tau) * dilated interior, interior)
  w_c  critical points: (|grad psi_hat| a / dpsi)^2 at the EFIT O- and X-points and
       ((psi_hat(x_X) - psi_b) / dpsi)^2
  w_s  scalar heads: Huber on standardised q95, betaN, R_axis, Z_axis, l_i (EFIT++ labels)
  w_t  temporal: squared error of frame-to-frame coefficient differences
Every epoch writes ``last.pt`` (model, optimiser, RNG states) so a run resumes after a crash.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import COMMON, REPO, assert_safe
from models.m9.network import _clip_features, augment_temporal, build_dualctx

GRID_R0, GRID_R1, GRID_Z0, GRID_Z1 = 0.06, 2.0, -2.0, 2.0
DR, DZ = (GRID_R1 - GRID_R0) / 64, (GRID_Z1 - GRID_Z0) / 64
SCAL = ("q95", "beta_n", "magnetic_axis_r", "magnetic_axis_z", "li")
TAU = 0.02
A_MINOR = 0.5


def load_scalars(keys, ids, fi) -> np.ndarray:
    lab = REPO / "data" / "fair_mast" / "labels"
    out = np.full((len(ids), len(SCAL)), np.nan, np.float32)
    for s, k in enumerate(keys):
        p = lab / f"{k}.npz"
        if not p.is_file():
            continue
        z = np.load(p)
        if "li" not in z.files:
            continue
        rows = np.flatnonzero(ids == s)
        f = fi[rows]
        n = len(z["time_ms"])
        okf = f < n
        for j, v in enumerate(SCAL):
            a = np.asarray(z[v], np.float64)
            out[rows[okf], j] = a[f[okf]]
    return out


def _sample(t, R, Zc):
    """Bilinear sample of (B,65,65) at points (B,) given in metres; NaN points -> 0 weight."""
    x = 2 * (R - GRID_R0) / (GRID_R1 - GRID_R0) - 1
    y = 2 * (Zc - GRID_Z0) / (GRID_Z1 - GRID_Z0) - 1
    g = torch.stack([x, y], -1).view(-1, 1, 1, 2)
    return F.grid_sample(t.unsqueeze(1), g, align_corners=True).view(-1)


def train(out_dir: Path, seed: int, w: dict, epochs: int = 50, cand_epochs=(), batch: int = 2048,
          device: str = "cuda", log=print) -> dict:
    out_dir = assert_safe(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "member.pt"
    if final.is_file():
        return torch.load(final, map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    dev = torch.device(device)
    d = np.load(COMMON / "data.npz")
    X, Z, ids, fi = d["X"], d["Z"], d["ids"], d["fi"]
    keys = [str(k) for k in d["keys"]]
    archval = set(str(k) for k in d["archval"])
    mu = torch.from_numpy(d["pca_mean"]).to(dev)
    C = torch.from_numpy(d["pca_components"]).to(dev)
    K, n_raw = C.shape[0], X.shape[1]
    val_shot = np.array([k in archval for k in keys])
    val_m = val_shot[ids]
    window, mode, clip = 9, "smoothdiff", 10.0
    X_aug = augment_temporal(X, ids, window, mode)
    del X
    X64 = np.asarray(X_aug[~val_m], np.float64)
    mean = np.nan_to_num(X64.mean(0)).astype(np.float32)
    std = X64.std(0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std).astype(np.float32)
    del X64
    xs = torch.from_numpy(np.ascontiguousarray(_clip_features(((X_aug - mean) / std).astype(np.float32), clip)))
    del X_aug
    zt = torch.from_numpy(Z).to(dev)
    use_route = any(w.get(k, 0) > 0 for k in ("w_b", "w_a", "w_c"))
    if use_route:
        r = np.load(COMMON / "route.npz")
        r_ok = torch.from_numpy(r["ok"]).to(dev)
        r_ax = torch.from_numpy(r["psi_ax"].astype(np.float32)).to(dev)
        r_b = torch.from_numpy(r["psi_b"].astype(np.float32)).to(dev)
        r_in = torch.from_numpy(np.unpackbits(r["inside"], axis=1)[:, :65 * 65].astype(bool))  # CPU
        r_o = torch.from_numpy(r["opt"].astype(np.float32)).to(dev)
        r_xu = torch.from_numpy(r["xu"].astype(np.float32)).to(dev)
        r_xl = torch.from_numpy(r["xl"].astype(np.float32)).to(dev)
        env = np.load(REPO / "fec_scoring" / "masks" / "mast_envelope.npz")["mask_coarse"].astype(bool)
        env_t = torch.from_numpy(env.reshape(-1)).to(dev)
    s_mean = s_std = None
    if w.get("w_s", 0) > 0:
        S = load_scalars(keys, ids, fi)
        tr = S[~val_m]
        s_mean = np.nanmean(tr, 0).astype(np.float32)
        s_std = np.maximum(np.nanstd(tr, 0), 1e-6).astype(np.float32)
        st = torch.from_numpy((S - s_mean) / s_std).to(dev)
        log(f"[m9x] scalar labels finite frac {np.isfinite(S).all(1).mean():.3f}")
    nxt = np.full(len(ids), -1, np.int64)
    same = ids[1:] == ids[:-1]
    nxt[:-1][same] = np.arange(1, len(ids))[same]
    nxt_t = torch.from_numpy(nxt)

    torch.manual_seed(int(seed))
    model = build_dualctx(xs.shape[1], K, n_raw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    tr_i, va_i = np.flatnonzero(~val_m), np.flatnonzero(val_m)
    tr_shots = np.unique(ids[tr_i])
    order = np.argsort(ids[tr_i], kind="stable")
    tr_sorted = tr_i[order]
    starts = np.searchsorted(ids[tr_sorted], tr_shots)
    lens = np.diff(np.append(starts, len(tr_sorted)))
    n_steps = max(1, int(math.ceil(len(tr_i) / batch)))
    state = {"ep": 0, "hist": [], "terms": [], "best_val": 1e30, "best_ep": -1}
    last = out_dir / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
        log(f"[m9x] resumed {out_dir.name} at epoch {state['ep']}")
    for ep in range(state["ep"], epochs):
        model.train()
        acc = {k: 0.0 for k in ("mse", "b", "a", "c", "s", "t")}
        for _ in range(n_steps):
            sid = rng.integers(0, len(tr_shots), size=batch)
            idx = tr_sorted[starts[sid] + (rng.random(batch) * lens[sid]).astype(np.int64)].astype(np.int64)
            it = torch.from_numpy(idx)
            outs = model(xs[it].to(dev, non_blocking=True))
            zh = outs[0]
            zi = zt[it.to(dev)]
            loss = ((zh - zi) ** 2).mean()
            acc["mse"] += float(loss.detach())
            if use_route:
                ig = it.to(dev)
                ok = r_ok[ig]
                if ok.any():
                    zh_o, zi_o, ig_o = zh[ok], zi[ok], ig[ok]
                    ph = zh_o @ C + mu
                    pt = zi_o @ C + mu
                    pax, pb = r_ax[ig_o][:, None], r_b[ig_o][:, None]
                    dps = (pax - pb).clamp_min(1e-6)
                    if w.get("w_b", 0) > 0:
                        band = ((((pax - pt) / dps) - 1).abs() < 0.1) & env_t[None, :]
                        lb = (band * (ph - pt) ** 2).sum(1).mean() / K
                        loss = loss + w["w_b"] * lb; acc["b"] += float(lb)
                    if w.get("w_a", 0) > 0:
                        tin = r_in[ig_o.cpu()].to(dev).float()
                        reg = F.max_pool2d(tin.view(-1, 1, 65, 65), 7, 1, 3).view(-1, 65 * 65)
                        sh = torch.sigmoid((1 - (pax - ph) / dps) / TAU) * reg
                        inter = (sh * tin).sum(1)
                        union = (sh + tin - sh * tin).sum(1).clamp_min(1.0)
                        la = (1 - inter / union).mean()
                        loss = loss + w["w_a"] * la; acc["a"] += float(la)
                    if w.get("w_c", 0) > 0:
                        g = ph.view(-1, 65, 65)
                        gz, gr = torch.gradient(g, spacing=(DZ, DR), dim=(1, 2))
                        gm = (gz ** 2 + gr ** 2) * (A_MINOR / dps.view(-1, 1, 1)) ** 2
                        terms = []
                        for P in (r_o[ig_o], r_xu[ig_o], r_xl[ig_o]):
                            fin = torch.isfinite(P).all(1)
                            if fin.any():
                                terms.append(_sample(gm[fin], P[fin, 0], P[fin, 1]))
                        for P in (r_xu[ig_o], r_xl[ig_o]):
                            fin = torch.isfinite(P).all(1)
                            if fin.any():
                                v = _sample(g[fin], P[fin, 0], P[fin, 1])
                                terms.append(((v - pb.view(-1)[fin]) / dps.view(-1)[fin]) ** 2)
                        lc = torch.cat(terms).mean() if terms else zh.sum() * 0
                        loss = loss + w["w_c"] * lc; acc["c"] += float(lc)
            if w.get("w_s", 0) > 0:
                tgt = st[it.to(dev)]
                pred = torch.stack(outs[1:6], 1)
                fin = torch.isfinite(tgt)
                ls = F.huber_loss(pred[fin], tgt[fin], delta=1.0) if fin.any() else zh.sum() * 0
                loss = loss + w["w_s"] * ls; acc["s"] += float(ls)
            if w.get("w_t", 0) > 0:
                nx = nxt_t[it]
                v = nx >= 0
                if v.any():
                    zh2 = model(xs[nx[v]].to(dev, non_blocking=True))[0]
                    vd = v.to(dev)
                    lt = (((zh2 - zh[vd]) - (zt[nx[v].to(dev)] - zi[vd])) ** 2).mean()
                    loss = loss + w["w_t"] * lt; acc["t"] += float(lt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pz = torch.cat([model(xs[va_i][s:s + 4096].to(dev))[0].float().cpu()
                            for s in range(0, len(va_i), 4096)]).numpy()
        val = float(((pz - Z[va_i]) ** 2).mean())
        state["hist"].append(val)
        state["terms"].append({k: v / n_steps for k, v in acc.items()})
        if val < state["best_val"]:
            state["best_val"], state["best_ep"] = val, ep
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "best_mse_state.pt")
        if ep in cand_epochs:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / f"cand_ep{ep:02d}.pt")
        state["ep"] = ep + 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "state": state}, out_dir / "last.part.pt")
        (out_dir / "last.part.pt").replace(last)
        log(f"[m9x] {out_dir.name} ep {ep:3d} val {val:.4e} terms "
            + " ".join(f"{k}={v / n_steps:.3g}" for k, v in acc.items() if v))
    meta = {"feature_mean": mean, "feature_std": std, "n_raw_features": int(n_raw),
            "n_features": int(xs.shape[1]), "n_pca": int(K), "pca_mean": d["pca_mean"],
            "pca_components": d["pca_components"], "arch": "dualctx", "temporal_window": window,
            "temporal_mode": mode, "feature_clip": clip, "init_seed": int(seed), "weights": w,
            "scalar_mean": s_mean, "scalar_std": s_std, "val_history": state["hist"], "terms": state["terms"],
            "best_mse_epoch": state["best_ep"], "best_val_mse": state["best_val"], "epochs": epochs,
            "train_seconds": round(time.perf_counter() - t0, 1)}
    torch.save(meta, out_dir / "meta.pt")
    return meta


def finalize(out_dir: Path, state_file: str, label: str) -> Path:
    """Write member.pt (M9 checkpoint layout) from a chosen state."""
    meta = torch.load(out_dir / "meta.pt", map_location="cpu", weights_only=False)
    ck = dict(meta)
    ck["model_state"] = torch.load(out_dir / state_file, map_location="cpu", weights_only=False)
    ck["selected"] = label
    torch.save(ck, out_dir / "member.pt")
    return out_dir / "member.pt"


class Ensemble:
    def __init__(self, member_files, device: str | None = None):
        from models.m9.features import EDGE_START, N_FEATURES, build_one, shot_from_mast_row
        self._feat = (EDGE_START, N_FEATURES, build_one, shot_from_mast_row)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members = []
        for f in member_files:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build_dualctx(int(ck["n_features"]), int(ck["n_pca"]), int(ck["n_raw_features"]))
            m.load_state_dict(ck["model_state"])
            self.members.append((m.to(self.dev).eval(), ck))

    @torch.no_grad()
    def predict(self, row: dict):
        e0, nf, build_one, smr = self._feat
        X, _ = build_one(smr(row))
        X = np.array(X, copy=True)
        X[:, e0:nf] = 0.0
        ids = np.zeros(len(X), np.int32)
        psis, heads = [], []
        for m, ck in self.members:
            Xa = augment_temporal(X, ids, int(ck["temporal_window"]), ck["temporal_mode"])
            Xp = _clip_features(((Xa - ck["feature_mean"]) / ck["feature_std"]).astype(np.float32),
                                float(ck["feature_clip"]))
            o = m(torch.from_numpy(np.ascontiguousarray(Xp)).to(self.dev))
            z = o[0].float().cpu().numpy()
            psis.append((z @ ck["pca_components"] + ck["pca_mean"]).reshape(-1, 65, 65).astype(np.float64))
            if ck.get("scalar_mean") is not None:
                h = torch.stack(o[1:6], 1).float().cpu().numpy()
                heads.append(h * ck["scalar_std"] + ck["scalar_mean"])
        return np.mean(psis, 0), (np.mean(heads, 0) if heads else None)


def write_meta(out: Path, d: dict) -> None:
    out.write_text(json.dumps(d, indent=1, default=str) + "\n")
