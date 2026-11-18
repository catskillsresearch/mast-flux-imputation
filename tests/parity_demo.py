"""Parity of the ported OSF and M9 against reference outputs of the original code.

The reference npz files are produced by running the ORIGINAL implementation on the 3 demo shots
(tests/reference_original.py); pass their directory as argv[1].

    python tests/parity_demo.py /tmp/mfi_work/ref
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.io import row_from_parquet  # noqa: E402
from models.osf import plasma_green, solve_shot  # noqa: E402
from models.m9 import M9Ensemble  # noqa: E402
from pipeline.paths import fec_demo_dir, m9_weights_dir  # noqa: E402

ref = Path(sys.argv[1])
ens = M9Ensemble(m9_weights_dir())
green = plasma_green(np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65))
worst = {}
for p in sorted(fec_demo_dir().glob("mast_shot_*.parquet")):
    r = np.load(ref / f"{p.stem}.npz")
    row = row_from_parquet(p)
    pack = solve_shot(row, green=green)
    m9 = ens.predict(row)
    d = {
        "osf_psi": float(np.max(np.abs(np.asarray(pack["psirz"]) - r["osf_psi"]))),
        "osf_q95": float(np.nanmax(np.abs(np.asarray(pack["q95"]) - r["osf_q"]))),
        "osf_betaN": float(np.nanmax(np.abs(np.asarray(pack["betaN"]) - r["osf_b"]))),
        "m9_psi": float(np.max(np.abs(m9 - r["m9"]))),
    }
    print(p.stem, d)
    for k, v in d.items():
        worst[k] = max(worst.get(k, 0.0), v)
print("max |diff|:", worst)
