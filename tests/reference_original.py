"""Reference outputs from the ORIGINAL catskills_fec implementation (for tests/parity_demo.py).

Run with the original repository on PYTHONPATH, bytecode off and a scratch lab root so nothing is
written into it or onto the lab drive:

    PYTHONDONTWRITEBYTECODE=1 CATSKILLS_FUSION_LAB=/tmp/fakelab PYTHONPATH=/path/to/catskills_fec \
    M9_WEIGHTS_DIR=models/m9/weights python tests/reference_original.py /tmp/ref data/fec/demo/*.parquet
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from catskills.coeff_cache import reconstruct
from catskills.mast_cache import EDGE_START, N_FEATURES, build_one, shot_from_mast_row
from catskills.pinn_charcoal import build_trunk
from catskills.run_crc_feq_pinn_mast import _infer_z, _prepare
from osf.mast_12_osf_0_0_12 import solve_shot
from osf.osf_0_0_1 import plasma_green


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(os.environ.get("M9_WEIGHTS_DIR", "models/m9/weights"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for tag in ("M9_win50_i101", "M9_win50_i202", "M9_win50_i303"):
        ck = torch.load(weights / f"CRC_FEQ_mast_{tag}.pt", map_location="cpu", weights_only=False)
        m = build_trunk("dualctx", int(ck["n_features"]), int(ck["n_pca"]), n_split=int(ck["n_raw_features"]),
                        trunk_width=512, trunk_blocks=4, ctx_width=512, ctx_blocks=3, trunk_dropout=0.05,
                        activation="relu").to(dev)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append((m, ck))
    green = plasma_green(np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65))
    for p in sys.argv[2:]:
        tbl = pq.read_table(p)
        row = {c: tbl[c][0].as_py() for c in tbl.column_names}
        X, _ = build_one(shot_from_mast_row(row))
        X = np.array(X, copy=True)
        X[:, EDGE_START:N_FEATURES] = 0.0
        mem = []
        for m, ck in models:
            Xp = _prepare(X, np.zeros(len(X), np.int32), int(ck["n_raw_features"]), ck["feature_mean"],
                          ck["feature_std"], ck["temporal_window"], ck["temporal_mode"], ck["feature_clip"])
            z = _infer_z(m, torch.from_numpy(np.ascontiguousarray(Xp)), dev)
            mem.append(np.asarray(reconstruct(z, ck["pca_mean"], ck["pca_components"]), np.float64))
        pack = solve_shot(row, green=green)
        out = out_dir / (Path(p).stem + ".npz")
        np.savez(out, m9=np.mean(mem, 0), osf_psi=pack["psirz"], osf_q=pack["q95"], osf_b=pack["betaN"])
        print(out)


if __name__ == "__main__":
    main()
