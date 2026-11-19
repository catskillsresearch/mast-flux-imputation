"""Agreement of the E6 soft (differentiable) read-outs with the scorer's extractors.

On 1500 archval100 frames (seed 0): scorer f_j (fec_scoring.derive via pipeline.audit) and the soft
read-outs on the EFIT psi (PCA reconstruction) and on the E4 ensemble's predicted psi. Reports the
Pearson correlation of the levels (truth) and of the errors (prediction - truth), which is what the
curve-distance loss acts on. Writes outputs/experiments/E6/soft_check.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from experiments.common import EXP
import os
import experiments.e6 as E
from experiments.e6 import Data, member_outputs
from pipeline import audit as A


def main():
    dev = torch.device("cuda")
    D = Data(dev, need_train_features=False)
    rng = np.random.default_rng(0)
    okv = np.flatnonzero(D.r_ok[D.v100])
    pick = np.sort(rng.choice(okv, size=min(1500, len(okv)), replace=False))
    idx = D.v100[pick]
    files = json.loads((EXP / "E4" / "members.json").read_text())["files"]
    zs = []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        zs.append(member_outputs(ck, D)[0])
    zp = torch.stack(zs).mean(0)[torch.from_numpy(pick).to(dev)]
    zt = torch.from_numpy(D.Z[idx]).to(dev)
    def soft(dil, text, tau, xt):
        E.DIL, E.T_EXT, E.TAU_S, E.XPT_TRI = dil, text, tau, xt
        with torch.no_grad():
            pax, pb = D.pab(idx)
            _, reg = D.region(idx)
            st = E.soft_feats(zt @ D.C + D.mu, pax, pb, reg, D.G, *D.xpts(idx))[0].cpu().numpy()
            sp = E.soft_feats(zp @ D.C + D.mu, pax, pb, reg, D.G, *D.xpts(idx))[0].cpu().numpy()
        return st, sp
    base = (E.DIL, E.T_EXT, E.TAU_S, E.XPT_TRI)
    settings = [base] + ([]
                         if os.environ.get("SOFT_SWEEP") else [])
    psit = (zt @ D.C + D.mu).cpu().numpy().reshape(-1, 65, 65).astype(np.float64)
    psip = (zp @ D.C + D.mu).cpu().numpy().reshape(-1, 65, 65).astype(np.float64)
    names = ["kappa", "tri_top", "tri_bot", "volume", "R_axis", "Z_axis"]
    ft = np.full((len(idx), 6), np.nan)
    fp = np.full((len(idx), 6), np.nan)
    for i in range(len(idx)):
        for arr, out in ((psit, ft), (psip, fp)):
            try:
                c = A.lcfs(arr[i])
                v = A.derive_frame(arr[i], A.R, A.Z, A.MACHINE, A.MASK, A.MASK_F, contour=c, with_li=False)
                out[i] = [v[n] for n in names]
            except Exception:  # noqa: BLE001
                pass
    res = {"n_frames": int(len(idx)), "settings": {}}
    for setting in settings:
        st, sp = soft(*setting)
        out = {}
        for j, n in enumerate(names):
            m = np.isfinite(ft[:, j]) & np.isfinite(fp[:, j])
            r_lvl = float(np.corrcoef(ft[m, j], st[m, j])[0, 1])
            r_err = float(np.corrcoef(fp[m, j] - ft[m, j], sp[m, j] - st[m, j])[0, 1])
            out[n] = {"n": int(m.sum()), "r_level": round(r_lvl, 4), "r_error": round(r_err, 4)}
        res["settings"][str(setting)] = out
        print(setting, " ".join(f"{n}:{v['r_level']:.3f}/{v['r_error']:.3f}" for n, v in out.items()), flush=True)
    res["used"] = res["settings"][str(base)]
    (EXP / "E6" / "soft_check.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
