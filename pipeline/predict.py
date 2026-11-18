#!/usr/bin/env python3
"""Impute MAST poloidal flux psi(R,Z), q95 and betaN without magnetic-probe / flux-loop inputs.

Modes
-----
``osf``     psi, q95, betaN all from the one-shot forward Grad-Shafranov solver
            ``mast_12_osf_0.0.12`` (no data-driven component).
``hybrid``  the shipped Phase-2 Final MAST recipe: psi = mean of the three M9_win50 members;
            q95, betaN = OSF read-outs through the post-hoc map
            ``q <- clip(q * 1.062242 / 1.20, 2, 25)``, ``b <- clip(b * 1.078508, 0.2, 5)``.

Output: ``<out>.npz`` in the FEC submission layout (``shot_XXXX_psirz`` (T,65,65),
``shot_XXXX_q95`` (T,), ``shot_XXXX_betaN`` (T,), XXXX = index in the split order) plus
``<out>.json`` with the shot keys, exclusions, weights checksums and timing.

    python -m pipeline.predict --split holdout --mode hybrid
    python -m pipeline.predict --split public_test --mode osf --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.datasets import SPLITS, load_shot, load_split, split_keys  # noqa: E402
from pipeline.paths import assert_not_on_lab, m9_weights_dir, output_dir  # noqa: E402

_GREEN = None


def _init_osf_worker() -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    global _GREEN
    from models.osf import plasma_green

    _GREEN = plasma_green(np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65))


def _osf_task(args: tuple[str, str, str]) -> tuple[str, float]:
    """Solve one shot with OSF and cache {psirz, q95, betaN} to ``cache_path``."""
    split, key, cache_path = args
    cp = Path(cache_path)
    if cp.is_file():
        return key, 0.0
    from models.osf import solve_shot

    t0 = time.time()
    shot = load_shot(split, key)
    pack = solve_shot(shot.row, green=_GREEN)
    tmp = cp.with_name(cp.stem + ".part.npz")
    np.savez(tmp, psirz=np.asarray(pack["psirz"], np.float64), q95=np.asarray(pack["q95"], np.float64),
             betaN=np.asarray(pack["betaN"], np.float64))
    tmp.replace(cp)
    return key, time.time() - t0


def run_osf(split: str, keys: list[str], cache_dir: Path, workers: int) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(split, k, str(cache_dir / f"{k}.npz")) for k in keys]
    todo = [j for j in jobs if not Path(j[2]).is_file()]
    print(f"[osf] {len(keys)} shots ({len(keys) - len(todo)} cached) workers={workers}", flush=True)
    if todo:
        if workers <= 1:
            _init_osf_worker()
            for i, j in enumerate(todo):
                k, dt = _osf_task(j)
                print(f"  [osf] {i + 1}/{len(todo)} {k} {dt:.1f}s", flush=True)
        else:
            import multiprocessing as mp

            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                     initializer=_init_osf_worker) as ex:
                for i, (k, dt) in enumerate(ex.map(_osf_task, todo, chunksize=1)):
                    print(f"  [osf] {i + 1}/{len(todo)} {k} {dt:.1f}s", flush=True)
    out = {}
    for k in keys:
        z = np.load(cache_dir / f"{k}.npz")
        out[k] = {n: z[n] for n in ("psirz", "q95", "betaN")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=SPLITS, default="holdout")
    ap.add_argument("--mode", choices=("osf", "hybrid"), default="hybrid")
    ap.add_argument("--out", type=Path, default=None, help="default outputs/<split>_<mode>.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="first N shots of the split (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--psi-dtype", choices=("float32", "float16"), default="float32",
                    help="float16 reproduces the shipped submission's storage precision")
    a = ap.parse_args()

    outdir = assert_not_on_lab(output_dir())
    out = assert_not_on_lab(a.out or outdir / f"{a.split}_{a.mode}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shots, excluded = load_split(a.split, a.limit)
    keys = [s.key for s in shots]
    print(f"[predict] split={a.split} mode={a.mode} shots={len(keys)} excluded={len(excluded)} "
          f"frames={sum(s.T for s in shots)}", flush=True)
    del shots

    osf = run_osf(a.split, keys, outdir / "cache" / "osf" / a.split, a.workers)
    meta: dict = {"split": a.split, "mode": a.mode, "keys": keys, "excluded_no_diverted_frames": excluded,
                  "osf_model": "mast_12_osf_0.0.12", "psi_dtype": a.psi_dtype}

    if a.mode == "hybrid":
        from models.m9 import M9Ensemble
        from models.osf import apply_final_scalars
        from models.osf.calibration import FINAL_SCALARS

        ens = M9Ensemble(m9_weights_dir(), device=a.device)
        meta["m9_members"] = ens.meta
        meta["final_scalars"] = FINAL_SCALARS

    arrays: dict[str, np.ndarray] = {}
    n_frames = 0
    for i, k in enumerate(keys):
        o = osf[k]
        if a.mode == "osf":
            psi, q, b = o["psirz"], o["q95"], o["betaN"]
        else:
            shot = load_shot(a.split, k)
            psi = ens.predict(shot.row)
            q, b = apply_final_scalars(o["q95"], o["betaN"])
            if psi.shape[0] != len(q):
                raise RuntimeError(f"{k}: M9 {psi.shape[0]} frames vs OSF {len(q)}")
        arrays[f"shot_{i:04d}_psirz"] = np.asarray(psi, dtype=a.psi_dtype)
        arrays[f"shot_{i:04d}_q95"] = np.asarray(q, np.float32)
        arrays[f"shot_{i:04d}_betaN"] = np.asarray(b, np.float32)
        n_frames += int(np.asarray(psi).shape[0])
    np.savez_compressed(out, **arrays)
    meta.update({"n_shots": len(keys), "n_frames": n_frames, "seconds": round(time.time() - t0, 1),
                 "npz": str(out)})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(f"[predict] wrote {out} ({len(keys)} shots, {n_frames} frames, {meta['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
