#!/usr/bin/env python3
"""Helpers for the Makefile workflow (paths from the environment the Makefile exports).

    status <stage> <state>   update $MFI_WORK/STATUS.json (stage, state, file counts, training stages)
    diskcheck --min-gb N     refuse to start when the work disk has less than N GB free
    shotlist / count / keys-json
    features                 fresh features vs the lab-drive cache (read-only), per shot and
                             column group, with time-base offsets and scale ratios
    smoke                    current model on the freshly built holdout and validation rows
    compare                  from-scratch model vs the current model (all score terms, curve distances)

All outputs go to $MFI_WORK (git-ignored); the lab drive is only read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = Path(os.environ.get("MFI_WORK", REPO / "work" / "xcheck"))
GROUPS = {"currents": (0, 14), "core_thomson": (14, 276), "edge_thomson": (276, 308)}
TERMS = ("S", "R2_psi", "R2_q95", "R2_betaN", "R2_qb", "one_minus_D_LCFS", "Consistency", "R2_R_axis",
         "R2_Z_axis", "R2_kappa", "R2_tri_top", "R2_tri_bot", "R2_volume", "R2_li")
CURVES = ("mean_hausdorff_mm", "p95_hausdorff_mm", "mean_area_symdiff_rel", "mean_opoint_disp_mm",
          "mean_xpt_disp_mm", "rate_topology_flip", "rate_jump", "mae_q95", "mae_betaN", "mae_kappa", "mae_li")
N_EXPECTED = 3738


def _safe(p: Path) -> Path:
    p = Path(p).resolve()
    if str(p).startswith("/media/catskills/CRC_FUSION_LAB"):
        raise RuntimeError(f"refusing to write on the lab drive: {p}")
    return p


def wj(p: Path, d) -> None:
    p = _safe(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=float) + "\n")
    tmp.replace(p)


def rj(p):
    return json.loads(Path(p).read_text())


def _count(d: Path, pat: str) -> int:
    return sum(1 for _ in d.glob(pat)) if d.is_dir() else 0


def status(stage: str | None, state: str | None) -> dict:
    f = WORK / "STATUS.json"
    st = rj(f) if f.is_file() else {"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if stage:
        rec = st["stages"].setdefault(stage, {})
        rec["state"] = state
        rec[state] = now
        if state == "done" and "running" in rec:
            t0 = time.mktime(time.strptime(rec["running"][:19], "%Y-%m-%d %H:%M:%S"))
            rec["seconds"] = round(time.time() - t0)
        if state == "running":
            st["stage"] = stage
    st["progress"] = {
        "parquet": f"{_count(WORK / 'parquet', 'mast_shot_*.parquet')}/{N_EXPECTED}",
        "labels": f"{_count(WORK / 'labels', 'mast_shot_*.npz')}/{N_EXPECTED}",
        "cache": f"{_count(WORK / 'cache', 'mast_shot_*.npz')}/{N_EXPECTED}",
    }
    e6 = WORK / "outputs" / "experiments" / "E6" / "STATUS.json"
    if e6.is_file():
        st["training"] = {k: v.get("state") for k, v in rj(e6).get("stages", {}).items()}
    du = shutil.disk_usage(WORK if WORK.exists() else REPO)
    st["disk_free_gb"] = round(du.free / 1e9, 1)
    st["updated"] = now
    wj(f, st)
    return st


def cmd_diskcheck(a):
    WORK.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORK).free / 1e9
    print(f"[xcheck] {free:.0f} GB free at {WORK} (need {a.min_gb})")
    if free < a.min_gb:
        raise SystemExit(f"not enough free disk: {free:.0f} GB < {a.min_gb} GB")


def cmd_shotlist(a):
    shots = [int(x) for x in Path(a.pool).read_text().split()]
    wj(WORK / "shotlist.json", {"pool": str(a.pool), "n": len(shots), "shots": shots})
    print(f"[xcheck] {len(shots)} shots")


def cmd_count(a):
    st = status(None, None)
    print(json.dumps(st["progress"]))


def cmd_keys_json(a):
    keys = sorted(p.stem for p in (WORK / "parquet").glob("mast_shot_*.parquet"))
    wj(Path(a.out), {"keys": keys})


# ------------------------------------------------------------------------------- feature diff
def _tcols(path: Path) -> dict:
    import pyarrow.parquet as pq
    cols = ("efit_times", "thomson_core_times", "thomson_edge_times")
    t = pq.read_table(path, columns=[c for c in cols if c in pq.read_schema(path).names]).to_pylist()[0]
    return {c: np.asarray(t.get(c) if t.get(c) is not None else [], np.float64) for c in cols}


def _timebase(fresh: Path, lab: Path) -> dict:
    """Per clock: lengths, offset (median difference) and scale ratio (span ratio) fresh vs lab."""
    out = {}
    try:
        a, b = _tcols(fresh), _tcols(lab)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    for c in a:
        x, y = a[c], b[c]
        r = {"n_fresh": int(x.size), "n_lab": int(y.size)}
        if x.size and y.size and x.size == y.size:
            r["offset_ms"] = float(np.median(x - y))
            sx, sy = float(np.ptp(x)), float(np.ptp(y))
            r["scale"] = sx / sy if sy else None
            r["max_abs_ms"] = float(np.max(np.abs(x - y)))
        out[c] = r
    return out


def _diff_one(args):
    key, fresh_cache, lab_cache, fresh_pq, lab_pq = args
    f, l = Path(fresh_cache) / f"{key}.npz", Path(lab_cache) / f"{key}.npz"
    rec = {"key": key}
    if not f.is_file() or not l.is_file():
        rec["missing"] = {"fresh": f.is_file(), "lab": l.is_file()}
        return rec
    with np.load(f) as d:
        X, Y = d["X"], d["Y"]
    with np.load(l) as d:
        Xl, Yl = d["X"], d["Y"]
    if X.shape != Xl.shape or Y.shape != Yl.shape:
        rec["shape"] = {"fresh": [list(X.shape), list(Y.shape)], "lab": [list(Xl.shape), list(Yl.shape)]}
    else:
        for g, (i, j) in GROUPS.items():
            a, b = X[:, i:j], Xl[:, i:j]
            same = np.array_equal(a, b, equal_nan=True)
            rec[g] = {"identical": bool(same)}
            if not same:
                dd = np.abs(np.nan_to_num(a.astype(np.float64)) - np.nan_to_num(b.astype(np.float64)))
                den = np.abs(np.nan_to_num(b.astype(np.float64))).max() or 1.0
                rec[g].update(max_abs=float(dd.max()), max_rel=float(dd.max() / den),
                              frac_frames=float((dd.max(1) > 0).mean()),
                              fresh_nonzero=float((a != 0).any(1).mean()), lab_nonzero=float((b != 0).any(1).mean()))
        rec["Y"] = {"identical": bool(np.array_equal(Y, Yl, equal_nan=True))}
    bad = "shape" in rec or any(not rec[g]["identical"] for g in (*GROUPS, "Y") if g in rec)
    rec["timebase"] = _timebase(Path(fresh_pq) / f"{key}.parquet", Path(lab_pq) / f"{key}.parquet")
    rec["identical"] = not bad
    return rec


def cmd_features(a):
    fresh = WORK / "cache"
    keys_f = {p.stem for p in fresh.glob("mast_shot_*.npz")}
    keys_l = {p.stem for p in Path(a.lab_cache).glob("mast_shot_*.npz")}
    keys = sorted(keys_f | keys_l)
    jobs = [(k, fresh, a.lab_cache, WORK / "parquet", a.lab_parquet) for k in keys]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_diff_one, jobs, chunksize=16))
    summ = {"fresh_shots": len(keys_f), "lab_shots": len(keys_l),
            "only_fresh": sorted(keys_f - keys_l), "only_lab": sorted(keys_l - keys_f),
            "identical_shots": sum(r.get("identical", False) for r in rows), "groups": {}}
    for g in (*GROUPS, "Y"):
        rr = [r for r in rows if g in r]
        diff = [r for r in rr if not r[g]["identical"]]
        summ["groups"][g] = {"compared": len(rr), "differ": len(diff),
                             "max_abs": max((r[g].get("max_abs", 0) for r in diff), default=0.0),
                             "max_rel": max((r[g].get("max_rel", 0) for r in diff), default=0.0)}
    offs = {}
    for r in rows:
        for c, t in (r.get("timebase") or {}).items():
            if isinstance(t, dict) and t.get("n_fresh") == 0 and t.get("n_lab") == 0:
                continue
            if isinstance(t, dict) and t.get("offset_ms") is not None:
                o = offs.setdefault(c, {"n": 0, "nonzero_offset": 0, "scale_not_1": 0, "len_mismatch": 0})
                o["n"] += 1
                o["nonzero_offset"] += abs(t["offset_ms"]) > 1e-6
                o["scale_not_1"] += t["scale"] is not None and abs(t["scale"] - 1) > 1e-9
                o["max_abs_ms"] = max(o.get("max_abs_ms", 0.0), t["max_abs_ms"])
            elif isinstance(t, dict) and t.get("n_fresh") != t.get("n_lab"):
                o = offs.setdefault(c, {"n": 0, "nonzero_offset": 0, "scale_not_1": 0, "len_mismatch": 0})
                o["n"] += 1
                o["len_mismatch"] += 1
    summ["timebase"] = offs
    # splits: the holdout recomputed from the fresh key set vs the tracked one
    hf, ht = WORK / "holdout_seed42.json", REPO / "data" / "fair_mast" / "holdout_seed42.json"
    if hf.is_file():
        summ["holdout_split_identical"] = rj(hf).get("sealed") == rj(ht).get("sealed")
    wj(WORK / "xcheck" / "features.json", {"summary": summ, "shots": rows})
    print(json.dumps({k: (len(v) if k.startswith("only_") else v) for k, v in summ.items()}, indent=1, default=float))


# ------------------------------------------------------------------------------- scoring checks
def _verdict(new: dict, ref: dict, s_drop: float, t_drop: float) -> dict:
    rows, fail = [], []
    for k in TERMS:
        if k in new and k in ref:
            d = float(new[k]) - float(ref[k])
            lim = s_drop if k == "S" else t_drop
            ok = d >= -lim
            rows.append({"term": k, "current": ref[k], "new": new[k], "diff": round(d, 6), "ok": ok})
            if not ok:
                fail.append(k)
    curves = [{"metric": k, "current": ref.get(k), "new": new.get(k)} for k in CURVES if k in new]
    return {"pass": not fail, "failed_terms": fail, "terms": rows, "curves": curves}


def _md(title: str, v: dict) -> str:
    L = [f"## {title}", "", f"pass: {v['pass']}  failed: {', '.join(v['failed_terms']) or 'none'}", "",
         "| term | current | new | diff |", "|---|---:|---:|---:|"]
    L += [f"| {r['term']} | {r['current']:.6f} | {r['new']:.6f} | {r['diff']:+.6f} |" for r in v["terms"]]
    L += ["", "| curve distance | current | new |", "|---|---:|---:|"]
    L += [f"| {r['metric']} | {r['current']:.5g} | {r['new']:.5g} |" for r in v["curves"] if r["current"] is not None]
    return "\n".join(L) + "\n"


def _attribution() -> list[str]:
    f = WORK / "xcheck" / "features.json"
    if not f.is_file():
        return ["no feature diff available"]
    s = rj(f)["summary"]
    out = []
    for g, r in s["groups"].items():
        if r["differ"]:
            out.append(f"{g}: {r['differ']}/{r['compared']} shots differ (max abs {r['max_abs']:.4g}, "
                       f"max rel {r['max_rel']:.3g})")
    for c, o in s.get("timebase", {}).items():
        if o["nonzero_offset"] or o["scale_not_1"] or o["len_mismatch"]:
            out.append(f"{c}: offset on {o['nonzero_offset']}, scale on {o['scale_not_1']}, "
                       f"length mismatch on {o['len_mismatch']} of {o['n']} shots")
    if s["only_lab"] or s["only_fresh"]:
        out.append(f"key sets differ: {len(s['only_lab'])} only in the lab cache, {len(s['only_fresh'])} only fresh")
    return out or ["no feature differences found; the cause is not the input data"]


def cmd_smoke(a):
    from experiments.driver import pred_audit
    files = rj(Path(a.ref_exp) / "select.json")["files"]
    res = {}
    for split in ("holdout", "demo"):
        new = pred_audit(split, f"xcheck_current_{split}_fresh", members=files, scalars="heads")
        ref = rj(Path(a.ref_audit) / f"exp_E6_{split}_hybrid" / "summary.json")
        res[split] = _verdict(new, ref, a.s_drop_max, a.term_drop_max)
    res["pass"] = res["holdout"]["pass"]
    res["attribution"] = _attribution()
    wj(WORK / "xcheck" / "smoke.json", res)
    (WORK / "xcheck" / "smoke.md").write_text(
        "# Current model on freshly built rows\n\n" + _md("holdout", res["holdout"]) + "\n" +
        _md("validation shots", res["demo"]) + "\n## feature differences\n\n" +
        "\n".join(f"- {x}" for x in res["attribution"]) + "\n")
    print((WORK / "xcheck" / "smoke.md").read_text())
    if not res["pass"]:
        raise SystemExit("smoke check failed: see xcheck/smoke.md (attribution listed)")


def cmd_compare(a):
    aud = WORK / "outputs" / "audit"
    res = {}
    for split in ("holdout", "demo"):
        new = rj(aud / f"exp_E6_{split}_hybrid" / "summary.json")
        ref = rj(Path(a.ref_audit) / f"exp_E6_{split}_hybrid" / "summary.json")
        res[split] = _verdict(new, ref, a.s_drop_max, a.term_drop_max)
    res["pass"] = res["holdout"]["pass"]
    res["attribution"] = _attribution() if not res["pass"] else []
    wj(WORK / "xcheck" / "compare.json", res)
    (WORK / "xcheck" / "compare.md").write_text(
        "# From-scratch model vs current model\n\n" + _md("holdout", res["holdout"]) + "\n" +
        _md("validation shots", res["demo"]) +
        ("\n## feature differences\n\n" + "\n".join(f"- {x}" for x in res["attribution"]) + "\n"
         if res["attribution"] else ""))
    print((WORK / "xcheck" / "compare.md").read_text())
    if not res["pass"]:
        raise SystemExit("from-scratch model is below the pass rule: see xcheck/compare.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("stage"); p.add_argument("state")
    p = sub.add_parser("refresh")
    p = sub.add_parser("diskcheck"); p.add_argument("--min-gb", type=float, default=60)
    p = sub.add_parser("shotlist"); p.add_argument("--pool", required=True)
    p = sub.add_parser("count"); p.add_argument("--what", default="parquet")
    p = sub.add_parser("keys-json"); p.add_argument("--out", required=True)
    p = sub.add_parser("features"); p.add_argument("--lab-cache", type=Path, required=True)
    p.add_argument("--lab-parquet", type=Path, required=True); p.add_argument("--workers", type=int, default=8)
    for name in ("smoke", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--ref-exp", type=Path, required=True); p.add_argument("--ref-audit", type=Path, required=True)
        p.add_argument("--s-drop-max", type=float, default=0.01); p.add_argument("--term-drop-max", type=float, default=0.10)
    a = ap.parse_args()
    if a.cmd == "status":
        status(a.stage, a.state)
    elif a.cmd == "refresh":
        status(None, None)
    else:
        {"diskcheck": cmd_diskcheck, "shotlist": cmd_shotlist, "count": cmd_count, "keys-json": cmd_keys_json,
         "features": cmd_features, "smoke": cmd_smoke, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
