"""E5: OSF read-out recalibration on FAIR-MAST training shots only (explicit optimiser).

OSF is solved once on ``osftrain200`` (200 training shots, FEC protocol cut, never archive-validation
or sealed). The q95 and betaN read-outs are then mapped by  q <- clip(a_q q_osf + b_q, 2, 25),
b <- clip(a_b b_osf + b_b, 0.2, 5), with (a, b) minimising the pooled squared error against the
EFIT++ labels (Nelder-Mead from the identity map; no leaderboard information). These maps replace
the shipped c*, s*. The 38 solver parameters are not refitted (cost reported from the timing).
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.common import EXP, assert_safe


def main(workers: int = 8) -> dict:
    from scipy.optimize import minimize
    from experiments.evaluate import osf_for
    from pipeline.datasets import load_shot, split_keys
    out = assert_safe(EXP / "E5")
    out.mkdir(parents=True, exist_ok=True)
    keys = split_keys("osftrain200")
    t0 = time.time()
    usable, gt_q, gt_b, os_q, os_b, frames = [], [], [], [], [], 0
    shots = {}
    for k in keys:
        s = load_shot("osftrain200", k)
        if s is not None:
            shots[k] = s.gt
            frames += len(s.gt["q95"])
    t_load = time.time() - t0
    t1 = time.time()
    osf = osf_for("osftrain200", list(shots), workers)
    t_osf = time.time() - t1
    for k, g in shots.items():
        gt_q.append(g["q95"]); gt_b.append(g["betaN"])
        os_q.append(osf[k]["q95"]); os_b.append(osf[k]["betaN"])
    gq, gb, oq, ob = (np.concatenate(v).astype(float) for v in (gt_q, gt_b, os_q, os_b))

    def fit(o, g, lo, hi):
        m = np.isfinite(o) & np.isfinite(g)
        o, g = o[m], g[m]
        sst = float(np.sum((g - g.mean()) ** 2))

        def sse(p):
            return float(np.sum((np.clip(p[0] * o + p[1], lo, hi) - g) ** 2))
        r0 = 1 - sse((1.0, 0.0)) / sst
        res = minimize(sse, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
        return res.x, r0, 1 - res.fun / sst, int(m.sum()), int(res.nit)

    (aq, bq), rq0, rq1, nq, itq = fit(oq, gq, 2.0, 25.0)
    (ab, bb), rb0, rb1, nb, itb = fit(ob, gb, 0.2, 5.0)
    maps = {"a_q": float(aq), "b_q": float(bq), "a_b": float(ab), "b_b": float(bb)}
    (out / "maps.json").write_text(json.dumps(maps, indent=1) + "\n")
    info = {"shots_with_diverted": len(shots), "frames": frames, "osf_seconds": round(t_osf, 1),
            "osf_workers": workers, "load_seconds": round(t_load, 1),
            "seconds_per_frame_per_worker": round(t_osf * workers / max(frames, 1), 3),
            "train_r2_q95_identity": rq0, "train_r2_q95_fit": rq1, "train_r2_betaN_identity": rb0,
            "train_r2_betaN_fit": rb1, "n_q": nq, "n_b": nb, "iters": [itq, itb], **maps}
    (out / "fit.json").write_text(json.dumps(info, indent=1) + "\n")
    print("[E5]", info, flush=True)
    return info


if __name__ == "__main__":
    main()
