"""Re-score predictions with catskills_fec's own MAST scorer (``catskills.mast_eval.score_holdout``,
which drives ``local_score.score_shots`` + ``fusion_scoring.metrics.finalize_machine``).

Run with the catskills_fec interpreter, read-only, no bytecode, lab root pointed at a scratch dir:

    PYTHONDONTWRITEBYTECODE=1 CATSKILLS_FUSION_LAB=/tmp/fakelab PYTHONPATH=<catskills_fec> \\
        <catskills_fec>/.venv/bin/python tests/crosscheck_catskills_scorer.py GT.npz PRED.npz OUT.json
"""
import json
import sys

import numpy as np
from catskills.mast_eval import MastShot, score_holdout


def main():
    gt_p, pred_p, out_p = sys.argv[1:4]
    g = np.load(gt_p)
    z = np.load(pred_p)
    keys = [str(k) for k in g["keys"]]
    shots, preds = [], []
    for i, k in enumerate(keys):
        psi = g[f"shot_{i:04d}_psirz"].astype(np.float64)
        T = psi.shape[0]
        shots.append(MastShot(key=k, vac=np.zeros_like(psi, np.float32), ip=np.zeros(T, np.float32), psi=psi,
                              q95=g[f"shot_{i:04d}_q95"].astype(np.float64),
                              betaN=g[f"shot_{i:04d}_betaN"].astype(np.float64), times=np.arange(T, dtype=np.float64)))
        preds.append({"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64),
                      "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
                      "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)})
    res = score_holdout(shots, preds, workers=int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    json.dump(res, open(out_p, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "r2_each", "r2_cons_each")}, default=float))


if __name__ == "__main__":
    main()
