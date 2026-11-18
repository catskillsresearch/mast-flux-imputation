#!/usr/bin/env python3
"""FAIR-MAST label coverage and format check against the FEC MAST rows (metadata + small arrays).

1. FEC demo shots (28348/28350/28351): every FEC ``efit_*`` label column is compared with the
   FAIR-MAST ``efm`` variable it corresponds to, on the FEC (diverted) frames.
2. A random sample of the M9 training pool (default 50 shots, seed 0): which ``efm`` label fields
   exist, their shapes / units / grid, and the fraction of diverted frames with finite values.
3. Ingest parquets of the same sample: EFIT grid, psi shape and clock against the FEC demo row.

Only scalar time series and metadata are read from S3 (no psi download). Writes
``outputs/compat/fair_mast_labels.json`` and prints markdown tables.

    python scripts/check_fair_mast_labels.py [--n 50] [--seed 0] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.paths import assert_not_on_lab, fair_mast_dir, fec_demo_dir, output_dir  # noqa: E402

ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast/level1/shots"
DEMO = (28348, 28350, 28351)
# FEC column -> efm variable (FEC scalars are per-frame; lcfs are padded polygons)
FEC_MAP = {"efit_q95": "q_95", "efit_beta_n": "betan", "efit_beta_p": "betap", "efit_li": "li",
           "efit_elongation": "elongation", "efit_r_axis": "magnetic_axis_r",
           "efit_z_axis": "magnetic_axis_z", "efit_geom_axis_r": "geom_axis_rc",
           "efit_geom_axis_z": "geom_axis_zc", "efit_psi_axis": "psi_axis",
           "efit_psi_boundary": "psi_boundary", "efit_lcfs_r": "lcfs_r", "efit_lcfs_z": "lcfs_z",
           "efit_lcfs_n": "lcfs_length"}
FIELDS = ["psirz", "q_95", "betan", "betap", "li", "elongation", "triang_upper", "triang_lower",
          "magnetic_axis_r", "magnetic_axis_z", "geom_axis_rc", "geom_axis_zc", "plasma_volume",
          "psi_axis", "psi_boundary", "lcfs_r", "lcfs_z", "lcfs_length", "minor_radius",
          "xpoint1_rc", "xpoint1_zc", "xpoint2_rc", "xpoint2_zc"]


def efm_info(fs, shot: int) -> dict:
    import xarray as xr
    try:
        efm = xr.open_zarr(fs.get_mapper(f"{BUCKET}/{shot}.zarr/efm"))
    except Exception as e:  # noqa: BLE001
        return {"shot": shot, "error": repr(e)[:200]}
    t = np.asarray(efm["time"], np.float64) * 1e3
    xr_ = [np.asarray(efm[v], np.float64) for v in ("xpoint1_rc", "xpoint2_rc") if v in efm]
    div = np.isfinite(np.vstack(xr_)).any(0) if xr_ else np.zeros(len(t), bool)
    out = {"shot": shot, "T": int(len(t)), "n_diverted": int(div.sum()), "t0_ms": float(t[0]),
           "dt_ms": float(np.median(np.diff(t))) if len(t) > 1 else None, "fields": {}}
    for v in FIELDS:
        if v not in efm:
            out["fields"][v] = {"present": False}
            continue
        da = efm[v]
        rec = {"present": True, "shape": list(da.shape), "dims": list(da.dims),
               "units": str(da.attrs.get("units", ""))}
        if v != "psirz":
            a = np.asarray(da, np.float64)
            a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a[:, None]
            fin = np.isfinite(a).any(1)
            rec["finite_frac_diverted"] = float(fin[div].mean()) if div.any() else None
        else:
            r = np.asarray(efm["profile_r"])
            rec["n_finite_R_cols"] = int(np.isfinite(np.asarray(da[0])).all(0).sum())
            rec["n_R_total"] = int(r.size)
        out["fields"][v] = rec
    return out


def demo_vs_efm(fs, shot: int) -> dict:
    import pyarrow.parquet as pq
    import xarray as xr
    row = pq.read_table(fec_demo_dir() / f"mast_shot_{shot}.parquet").to_pylist()[0]
    efm = xr.open_zarr(fs.get_mapper(f"{BUCKET}/{shot}.zarr/efm"))
    te = np.asarray(efm["time"], np.float64) * 1e3
    tf = np.asarray(row["efit_times"], np.float64)
    idx = np.array([int(np.argmin(np.abs(te - x))) for x in tf])
    out = {"shot": shot, "frames": int(len(tf)), "max_dt_ms": float(np.max(np.abs(te[idx] - tf)))}
    for col, v in FEC_MAP.items():
        if col not in row or row[col] is None:
            out[col] = "absent in FEC"
            continue
        f = np.asarray(row[col], np.float64)
        if v not in efm:
            out[col] = f"no efm {v}"
            continue
        e = np.asarray(efm[v], np.float64)[idx]
        if f.shape != e.shape:
            if f.ndim == 2 and e.ndim == 2:
                n = min(f.shape[1], e.shape[1])
                fe, ee = f[:, :n], e[:, :n]
                m = np.isfinite(fe) & np.isfinite(ee)
                out[col] = {"fec_shape": list(f.shape), "efm_shape": list(e.shape),
                            "max_abs_diff_common": float(np.max(np.abs(fe[m] - ee[m]))) if m.any() else None}
            else:
                out[col] = {"fec_shape": list(f.shape), "efm_shape": list(e.shape)}
            continue
        m = np.isfinite(f) & np.isfinite(e)
        out[col] = {"exact": bool(np.array_equal(f.astype(np.float32), e.astype(np.float32), equal_nan=True)),
                    "max_abs_diff": float(np.max(np.abs(f[m] - e[m]))) if m.any() else None}
    return out


def ingest_info(key: str, ref: dict) -> dict:
    import pyarrow.parquet as pq
    p = fair_mast_dir() / f"{key}.parquet"
    t = pq.read_table(p, columns=["efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time",
                                  "thomson_core_R", "coil_name"]).to_pylist()[0]
    md = pq.ParquetFile(p).metadata
    return {"key": key, "T": len(t["efit_times"]),
            "grid_R_equal_fec": bool(np.allclose(t["efit_grid_R"], ref["efit_grid_R"])),
            "grid_Z_equal_fec": bool(np.allclose(t["efit_grid_Z"], ref["efit_grid_Z"])),
            "mag_t0_ms": float(t["magnetics_time"][0]), "mag_n": len(t["magnetics_time"]),
            "core_R_n": len(t["thomson_core_R"]), "coil_table_equal_fec": t["coil_name"] == ref["coil_name"],
            "rows": md.num_rows}


def main() -> int:
    import pyarrow.parquet as pq
    import s3fs
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fs = s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT, asynchronous=False)
    keep = (ROOT / "data/fair_mast/mast_qc_keep.txt").read_text().split()
    keys = sorted(keep)
    nums = sorted(int(k.split("_")[-1]) for k in (p.stem for p in fair_mast_dir().glob("mast_shot_*.parquet")))
    sample = sorted(random.Random(a.seed).sample(keys, a.n))
    ref = pq.read_table(fec_demo_dir() / "mast_shot_28348.parquet",
                        columns=["efit_grid_R", "efit_grid_Z", "coil_name"]).to_pylist()[0]
    with ThreadPoolExecutor(a.threads) as ex:
        demo = list(ex.map(lambda s: demo_vs_efm(fs, s), DEMO))
        efm = list(ex.map(lambda k: efm_info(fs, int(k.split("_")[-1])), sample))
        ing = list(ex.map(lambda k: ingest_info(k, ref), sample))
    ok = [e for e in efm if "error" not in e]
    cov = {}
    for v in FIELDS:
        pres = [e["fields"][v]["present"] for e in ok]
        fins = [e["fields"][v].get("finite_frac_diverted") for e in ok if e["fields"][v]["present"]]
        fins = [x for x in fins if x is not None]
        units = sorted({e["fields"][v].get("units", "") for e in ok if e["fields"][v]["present"]})
        shapes = sorted({str(e["fields"][v].get("shape", [None])[1:]) for e in ok if e["fields"][v]["present"]})
        cov[v] = {"present_frac": float(np.mean(pres)) if pres else 0.0,
                  "mean_finite_frac_diverted": float(np.mean(fins)) if fins else None,
                  "units": units, "per_frame_shape": shapes}
    res = {"ingest_shot_range": [nums[0], nums[-1]], "ingest_n": len(nums), "sample": sample,
           "n_sample_ok": len(ok), "errors": [e for e in efm if "error" in e], "coverage": cov,
           "efm_clock": {"dt_ms_values": sorted({round(e["dt_ms"], 3) for e in ok if e["dt_ms"]}),
                         "frames_median": float(np.median([e["T"] for e in ok])),
                         "diverted_frac_median": float(np.median([e["n_diverted"] / e["T"] for e in ok]))},
           "psirz_grid": sorted({(e["fields"]["psirz"].get("n_finite_R_cols"), e["fields"]["psirz"].get("n_R_total"))
                                 for e in ok if e["fields"]["psirz"]["present"]}),
           "ingest": {"grid_equal_fec_frac": float(np.mean([i["grid_R_equal_fec"] and i["grid_Z_equal_fec"] for i in ing])),
                      "coil_table_equal_frac": float(np.mean([i["coil_table_equal_fec"] for i in ing])),
                      "mag_t0_ms": sorted({i["mag_t0_ms"] for i in ing}), "mag_n": sorted({i["mag_n"] for i in ing}),
                      "core_R_n": sorted({i["core_R_n"] for i in ing})},
           "demo_vs_efm": demo,
           "per_shot_efm": [{"shot": e["shot"], "T": e.get("T"), "n_diverted": e.get("n_diverted"),
                             "dt_ms": e.get("dt_ms"),
                             "psirz_shape": e["fields"]["psirz"].get("shape") if "fields" in e else None,
                             "has_q95": e["fields"]["q_95"]["present"] if "fields" in e else None,
                             "n_fields_present": sum(f["present"] for f in e["fields"].values()) if "fields" in e else 0}
                            for e in efm]}
    out = assert_not_on_lab(output_dir() / "compat")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fair_mast_labels.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"ingest shots {len(nums)} range {nums[0]}-{nums[-1]}; sample {len(ok)}/{a.n} ok")
    print("| efm field | present | finite on diverted frames | units | per-frame shape |\n|---|---|---|---|---|")
    for v, c in cov.items():
        f = c["mean_finite_frac_diverted"]
        print(f"| {v} | {c['present_frac']:.0%} | {'n/a' if f is None else f'{f:.1%}'} | {', '.join(c['units']) or '-'} | {', '.join(c['per_frame_shape'])} |")
    print(json.dumps({k: res[k] for k in ("efm_clock", "psirz_grid", "ingest")}, default=str))
    for d in demo:
        print(json.dumps(d, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
