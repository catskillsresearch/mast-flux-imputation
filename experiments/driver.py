"""E1-E5 driver: runs every stage in order, each in its own process with its own log, and records
progress in outputs/experiments/STATUS.json. Completed stages, members and audits are skipped on a
rerun, so the driver resumes after a crash. Rules are pre-declared in docs/TRAINING_REGIME.md.

    python -m experiments.driver            # run / resume everything
    python -m experiments.driver --stage E2 # one stage (used internally)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline.paths import output_dir
from experiments.common import COMMON, EXP, REPO, SPLITS_JSON, route_score, stage_done, status_update

PY = sys.executable
WORKERS = 8
STAGES = ("prep", "labels_fast", "route", "baseline", "E5", "E1", "E2", "E3", "E4", "final", "holdout")
LOGDIR = {"prep": "common", "labels_fast": "common", "route": "common", "baseline": "common", "final": "final",
          "holdout": "holdout"}
SWEEP_EPOCHS, FULL_EPOCHS = 10, 50
CAND_EPOCHS = (19, 29, 39, 44, 49)
SEEDS = (101, 202, 303)
SWEEPS = {
    "E2": [{"w_b": 1.0, "w_a": 0.0}, {"w_b": 3.0, "w_a": 0.0}, {"w_b": 1.0, "w_a": 3e-3}, {"w_b": 3.0, "w_a": 1e-2}],
    "E3": [{"w_s": 3e-4}, {"w_s": 3e-3}],
    "E4": [{"w_c": 1e-3, "w_t": 1.0}, {"w_c": 1e-2, "w_t": 1.0}],
}
SCALARS = {"E1": "osf_e5", "E2": "osf_e5", "E3": "heads", "E4": "heads"}
OLD_DEMO = {"hybrid": "demo_hybrid_shipped", "osf": "demo_osf_shipped"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=1, default=float) + "\n")


def pred_audit(split, tag, **kw):
    """Predict (if not already) and audit (if not already); returns the audit summary."""
    from experiments.evaluate import audit, predict_split
    summ = output_dir() / "audit" / tag / "summary.json"
    if summ.is_file():
        return rj(summ)
    npz = EXP / "preds" / f"{tag}.npz"
    if not npz.with_suffix(".json").is_file():
        predict_split(split, npz, workers=WORKERS, **kw)
    return audit(npz, tag, WORKERS)


def compare_md(old_tag, new_tag, out):
    from experiments.evaluate import compare_table
    t = compare_table(old_tag, new_tag)
    Path(out).write_text(t)
    log(t)
    return t


# ----------------------------------------------------------------------------------------- stages
def st_prep():
    from experiments import prep
    if not (COMMON / "data.npz").is_file():
        prep.main()


def _keys_json(name, keys):
    p = COMMON / f"keys_{name}.json"
    wj(p, {"keys": list(keys)})
    return p


def st_labels_fast():
    sp = rj(SPLITS_JSON)
    p = _keys_json("fast", sorted(set(sp["archval100"]) | set(sp["osftrain200"])))
    subprocess.run([PY, "data/fair_mast/fetch_efit_labels.py", "--keys-json", str(p), "--workers", "16"],
                   cwd=REPO, check=False)
    import numpy as np
    keys = [str(k) for k in np.load(COMMON / "data.npz")["keys"]]
    _keys_json("all", keys)
    _start_label_fetch()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return st != "Z"
    except OSError:
        return False


def _start_label_fetch():
    """Background EFIT++ label fetch for every contributing shot (resumable; cached files skipped)."""
    pa = COMMON / "keys_all.json"
    if not pa.is_file():
        return
    keys = rj(pa)["keys"]
    have = sum((REPO / "data/fair_mast/labels" / f"{k}.npz").is_file() for k in keys)
    pf = COMMON / "labels_all.pid.json"
    if have >= len(keys) or (pf.is_file() and _pid_alive(rj(pf)["pid"])):
        return
    lg = open(COMMON / "labels_all.log.txt", "a")
    pr = subprocess.Popen([PY, "data/fair_mast/fetch_efit_labels.py", "--keys-json", str(pa), "--workers", "16"],
                          cwd=REPO, stdout=lg, stderr=subprocess.STDOUT, start_new_session=False)
    wj(pf, {"pid": pr.pid, "started": time.time(), "have": have})
    log(f"background label fetch for {len(keys)} shots, pid {pr.pid}")


def st_route():
    from experiments import route_labels
    if not (COMMON / "route.npz").is_file():
        route_labels.main(WORKERS)


def st_baseline():
    from experiments.evaluate import osf_for
    from pipeline.datasets import load_split
    for split in ("archval100",):
        keys = [s.key for s in load_split(split)[0]]
        osf_for(split, keys, WORKERS)
    a = pred_audit("archval100", "exp_archval100_hybrid_shipped", shipped=True, scalars="osf_shipped")
    b = pred_audit("select30", "exp_select30_m9_shipped", shipped=True, scalars="const")
    c = pred_audit("archval100", "exp_archval100_osf_shipped", psi_from="osf", scalars="osf_raw")
    wj(COMMON / "baseline.json", {"archval100_hybrid": a, "select30_m9": b, "archval100_osf": c})


def st_E5():
    from experiments import osf_recal
    d = EXP / "E5"
    if not (d / "fit.json").is_file():
        osf_recal.main(WORKERS)
    fit = rj(d / "fit.json")
    new_demo = pred_audit("demo", "exp_E5_demo_osf", psi_from="osf", scalars="osf_e5")
    new_av = pred_audit("archval100", "exp_E5_archval100_osf", psi_from="osf", scalars="osf_e5")
    compare_md(OLD_DEMO["osf"], "exp_E5_demo_osf", d / "compare_demo.md")
    compare_md("exp_archval100_osf_shipped", "exp_E5_archval100_osf", d / "compare_archval100.md")
    wj(d / "result.json", {"fit": fit, "demo": new_demo, "archval100": new_av})


def _ref_select():
    return rj(REPO / "outputs/audit/exp_select30_m9_shipped/summary.json")


def _member_file(mdir: Path, state: str, label: str) -> Path:
    import torch
    f = mdir / f"cand_{label}.member.pt"
    if not f.is_file():
        ck = dict(torch.load(mdir / "meta.pt", map_location="cpu", weights_only=False))
        ck["model_state"] = torch.load(mdir / state, map_location="cpu", weights_only=False)
        ck["selected"] = label
        torch.save(ck, f)
    return f


def _train(mdir, seed, w, epochs, cands):
    from experiments.m9x import train
    return train(mdir, seed, w, epochs=epochs, cand_epochs=cands, log=log)


def _sweep(E, base_w):
    """Member 101, 10 epochs, epoch-9 state; returns chosen weights (persisted)."""
    d = EXP / E
    choice = d / "sweep_choice.json"
    if choice.is_file():
        return rj(choice)["w"]
    ref = _ref_select()
    rows = []
    for i, extra in enumerate(SWEEPS[E]):
        w = {**base_w, **extra}
        md = d / f"sweep{i}"
        _train(md, 101, w, SWEEP_EPOCHS, (SWEEP_EPOCHS - 1,))
        f = _member_file(md, f"cand_ep{SWEEP_EPOCHS - 1:02d}.pt", f"ep{SWEEP_EPOCHS - 1:02d}")
        sc = "heads" if w.get("w_s", 0) > 0 else "const"
        s = pred_audit("select30", f"exp_{E}_sweep{i}_select30", members=[f], scalars=sc)
        rows.append({"i": i, "w": w, "R": route_score(s, ref), "R2_qb": s.get("R2_qb"), "R2_psi": s.get("R2_psi"),
                     "S": s.get("S")})
        log(f"{E} sweep {i} {w} R={rows[-1]['R']:.4f} R2_qb={rows[-1]['R2_qb']}")
    if E == "E3":  # pre-declared: best R2_qb among settings with R within 5 % of the lowest R
        rmin = min(r["R"] for r in rows)
        pick = max((r for r in rows if r["R"] <= 1.05 * rmin), key=lambda r: r["R2_qb"])
    else:
        pick = min(rows, key=lambda r: r["R"])
    wj(choice, {"w": pick["w"], "rows": rows, "pick": pick["i"]})
    return pick["w"]


def _members(E, w, arm_b: bool):
    d = EXP / E
    ref = _ref_select()
    files, sel = [], []
    for seed in SEEDS:
        md = d / f"m{seed}"
        meta = _train(md, seed, w, FULL_EPOCHS, CAND_EPOCHS if arm_b else ())
        if not arm_b:  # arm A rule: best archive-validation coefficient MSE
            files.append(_member_file(md, "best_mse_state.pt", "best_mse"))
            sel.append({"seed": seed, "pick": "best_mse", "epoch": meta["best_mse_epoch"]})
            continue
        cands = [(f"cand_ep{e:02d}.pt", f"ep{e:02d}") for e in CAND_EPOCHS] + [("best_mse_state.pt", "best_mse")]
        rows = []
        for st, lab in cands:
            f = _member_file(md, st, lab)
            s = pred_audit("select30", f"exp_{E}_m{seed}_{lab}_select30", members=[f], scalars="const")
            rows.append({"label": lab, "file": str(f), "R": route_score(s, ref), "R2_psi": s["R2_psi"]})
        best = min(rows, key=lambda r: r["R"])
        files.append(Path(best["file"]))
        sel.append({"seed": seed, "pick": best["label"], "rows": rows,
                    "best_mse_epoch": meta["best_mse_epoch"]})
        log(f"{E} m{seed} pick {best['label']} R={best['R']:.4f}")
    wj(d / "members.json", {"files": [str(f) for f in files], "selection": sel, "w": w})
    return files


def _evaluate(E, files):
    d = EXP / E
    sc = SCALARS[E]
    demo = pred_audit("demo", f"exp_{E}_demo_hybrid", members=files, scalars=sc)
    av = pred_audit("archval100", f"exp_{E}_archval100_hybrid", members=files, scalars=sc)
    compare_md(OLD_DEMO["hybrid"], f"exp_{E}_demo_hybrid", d / "compare_demo.md")
    compare_md("exp_archval100_hybrid_shipped", f"exp_{E}_archval100_hybrid", d / "compare_archval100.md")
    base = rj(COMMON / "baseline.json")["archval100_hybrid"]
    res = {"demo": demo, "archval100": av, "R_archval100": route_score(av, base), "scalars": sc,
           "members": [str(f) for f in files]}
    wj(d / "result.json", res)
    log(f"{E} demo S={demo['S']:.6f} archval100 R={res['R_archval100']:.4f}")


def st_E1():
    _evaluate("E1", _members("E1", {}, arm_b=False))


def _wait_labels():
    p = COMMON / "labels_all.pid.json"
    if p.is_file():
        pid = rj(p)["pid"]
        while _pid_alive(pid):
            log("waiting for background label fetch ...")
            time.sleep(120)
    subprocess.run([PY, "data/fair_mast/fetch_efit_labels.py", "--keys-json", str(COMMON / "keys_all.json"),
                    "--workers", "16"], cwd=REPO, check=False, stdout=subprocess.DEVNULL)


def st_E2():
    w = _sweep("E2", {})
    _evaluate("E2", _members("E2", w, arm_b=True))


def st_E3():
    _wait_labels()
    w = _sweep("E3", rj(EXP / "E2" / "sweep_choice.json")["w"])
    _evaluate("E3", _members("E3", w, arm_b=True))


def st_E4():
    w = _sweep("E4", rj(EXP / "E3" / "sweep_choice.json")["w"])
    _evaluate("E4", _members("E4", w, arm_b=True))


def st_final():
    """Pre-declared: lowest R on archval100 among E2-E4 (reference shipped on archval100); tie -> demo S."""
    rows = []
    for E in ("E2", "E3", "E4"):
        r = rj(EXP / E / "result.json")
        rows.append({"E": E, "R": r["R_archval100"], "demo_S": r["demo"]["S"]})
    pick = min(rows, key=lambda r: (round(r["R"], 4), -r["demo_S"]))
    e1 = rj(EXP / "E1" / "result.json")
    wj(EXP / "final" / "pick.json", {"rows": rows, "pick": pick["E"], "E1_R_archval100": e1["R_archval100"]})
    log("final pick", pick)


def st_holdout():
    """One-time sealed-holdout check: shipped (existing audit), E1, the final pick, E5 OSF mode."""
    d = EXP / "holdout"
    if (d / "result.json").is_file():
        return
    pick = rj(EXP / "final" / "pick.json")["pick"]
    out = {"pick": pick}
    for E in ("E1", pick):
        files = rj(EXP / E / "members.json")["files"]
        out[E] = pred_audit("holdout", f"exp_{E}_holdout_hybrid", members=files, scalars=SCALARS[E])
        compare_md("holdout_hybrid_shipped", f"exp_{E}_holdout_hybrid", d / f"compare_{E}.md")
    out["E5"] = pred_audit("holdout", "exp_E5_holdout_osf", psi_from="osf", scalars="osf_e5")
    compare_md("holdout_osf_shipped", "exp_E5_holdout_osf", d / "compare_E5.md")
    compare_md("exp_E1_holdout_hybrid", f"exp_{pick}_holdout_hybrid", d / f"compare_E1_vs_{pick}.md")
    wj(d / "result.json", out)


FUNCS = {n: globals()[f"st_{n}"] for n in STAGES}


def run_one(name):
    t0 = time.time()
    status_update(name, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    try:
        FUNCS[name]()
    except BaseException as e:
        status_update(name, state="failed", error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
        raise
    status_update(name, state="done", seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--until", choices=STAGES)
    a = ap.parse_args()
    if a.stage:
        run_one(a.stage)
        return 0
    status_update("driver", state="running", driver_pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if stage_done("labels_fast"):
        _start_label_fetch()
    for name in STAGES:
        if stage_done(name):
            log(f"skip {name} (done)")
        else:
            ld = EXP / LOGDIR.get(name, name)
            ld.mkdir(parents=True, exist_ok=True)
            lf = ld / ("log.txt" if name not in ("prep", "labels_fast", "route", "baseline") else f"{name}.log.txt")
            log(f"start {name} -> {lf}")
            with open(lf, "a") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} start {name}\n")
                fh.flush()
                rc = subprocess.run([PY, "-m", "experiments.driver", "--stage", name], cwd=REPO, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                status_update("driver", state="failed", failed_stage=name)
                log(f"{name} failed rc={rc}; see {lf}")
                return rc
            log(f"done {name}")
        if a.until == name:
            break
    status_update("driver", state="done" if not a.until else "paused", finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
