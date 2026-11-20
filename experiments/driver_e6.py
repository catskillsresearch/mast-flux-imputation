"""E6 driver (curve-distance training) with stage skipping and resume.

    python -m experiments.driver_e6            # run / resume every stage not yet done
    python -m experiments.driver_e6 --stage m202

Stages: truth -> sweep -> m101 m202 m303 m404 m505 -> select -> final. Progress is in
outputs/experiments/E6/STATUS.json; a stage marked done is skipped, and a training stage resumes
from its last.pt. The archive objective uses archval100 (FAIR-MAST) only; the three validation shots
and the sealed holdout are touched once, in `final`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import EXP, REPO

E6 = EXP / "E6"
SEEDS = (101, 202, 303, 404, 505)
STAGES = ("truth", "sweep", *[f"m{s}" for s in SEEDS], "select", "final")
BASE_W = {"w_lcfs": 0.0, "w_b": 3.0, "w_tip": 10.0, "w_c": 1e-3, "w_t": 1.0, "w_s": 3e-3}
SWEEP = [0.01, 0.03, 0.1, 0.3]  # w_cd; w_lcfs = w_cd / 3
SWEEP_EPOCHS = 40
PY = sys.executable
STATUS = E6 / "STATUS.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=1, default=float) + "\n")


def status(stage, **kw):
    E6.mkdir(parents=True, exist_ok=True)
    st = rj(STATUS) if STATUS.is_file() else {"stages": {}}
    rec = st["stages"].get(stage, {})
    if kw.get("state") == "running":
        for k in ("error", "seconds", "finished", "failed_stage"):
            rec.pop(k, None)
    rec.update(kw)
    rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    st["stages"][stage] = rec
    st["updated"] = rec["updated"]
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str) + "\n")
    tmp.replace(STATUS)


def done(stage):
    return STATUS.is_file() and rj(STATUS)["stages"].get(stage, {}).get("state") == "done"


def _data(need_train=True):
    import torch
    from experiments.e6 import Data
    return Data(torch.device("cuda"), need_train_features=need_train, log=log)


def pred_audit(split, tag, files):
    from experiments.evaluate import pred_audit as pa
    return pa(split, tag, members=[str(f) for f in files], scalars="heads")


# Selection gate on archval100 S (workflow/config.mk sets MFI_SELECT_GATE_S to the same value).
SELECT_GATE_S = float(os.environ.get("MFI_SELECT_GATE_S", "0.934308"))


# ------------------------------------------------------------------------------------- stages
def st_truth():
    _data(need_train=False)


def st_sweep():
    from experiments.e6 import train
    ch = E6 / "sweep_choice.json"
    if ch.is_file():
        return
    D = _data()
    rows = []
    for i, wcd in enumerate(SWEEP):
        w = {**BASE_W, "w_cd": wcd, "w_lcfs": wcd / 3}
        meta = train(E6 / f"sweep{i}", 101, w, D, max_epochs=SWEEP_EPOCHS, min_epochs=SWEEP_EPOCHS, log=log)
        h = meta["history"]
        bestJ = min(r["J"] for r in h)
        bestM = min(r["mse"] for r in h)
        rows.append({"i": i, "w": w, "best_J": bestJ, "best_mse": bestM})
        log(f"sweep {i} w_cd={wcd} best J {bestJ:.5f} best mse {bestM:.3e}")
    mmin = min(r["best_mse"] for r in rows)
    ok = [r for r in rows if r["best_mse"] <= 1.10 * mmin]
    pick = min(ok, key=lambda r: r["best_J"])
    wj(ch, {"rule": "lowest best J among settings with best archval100 coefficient MSE within 10% of the lowest",
            "rows": rows, "pick": pick["i"], "w": pick["w"]})


def _st_member(seed):
    from experiments.e6 import train
    w = rj(E6 / "sweep_choice.json")["w"]
    D = _data()
    meta = train(E6 / f"m{seed}", seed, w, D, log=log)
    wj(E6 / f"m{seed}" / "summary.json", {k: meta.get(k) for k in ("stopped_epoch", "best_J", "best_J_epoch", "top", "rollbacks",
                                                              "best_mse_epoch", "train_seconds", "weights")})


def st_select():
    """Rank-k ensemble (k-th best EMA state by J of each seed), k = 1, 2, 3; take the first whose
    archval100 S reaches the gate; if none, the best-S one (reported)."""
    from experiments.e6 import ensemble_objective, member_file
    out = E6 / "select.json"
    prev = rj(out) if out.is_file() else {"rows": []}
    D = _data(need_train=False)
    gate = SELECT_GATE_S
    rows = prev["rows"]
    pick = None
    for k in range(3):
        if k < len(rows):
            r = rows[k]
        else:
            files = []
            for seed in SEEDS:
                md = E6 / f"m{seed}"
                top = rj(md / "summary.json")["top"]
                t = top[min(k, len(top) - 1)]
                files.append(str(member_file(md, t["file"], f"rank{k + 1}_ep{t['ep']:03d}")))
            ob = ensemble_objective(files, D)
            au = pred_audit("archval100", f"exp_E6_rank{k + 1}_archval100_hybrid", files)
            r = {"k": k + 1, "files": files, "objective": ob, "archval100": au}
            rows.append(r)
            wj(out, {"rows": rows, "gate_S": gate})
        log(f"select rank{k + 1}: J {r['objective']['J']:.5f} archival S {r['archval100']['S']} (gate {gate})")
        if r["archval100"]["S"] >= gate:
            pick = r
            break
    gated = pick is not None
    if pick is None:
        pick = max(rows, key=lambda r: r["archval100"]["S"])
    wj(out, {"rows": rows, "gate_S": gate, "gate_passed": gated, "pick": pick["k"], "files": pick["files"]})


def st_final():
    """One-time scoring of the selected ensemble: the three validation shots and the sealed holdout."""
    sel = rj(E6 / "select.json")
    files = sel["files"]
    res = {"pick_rank": sel["pick"]}
    res["demo"] = pred_audit("demo", "exp_E6_demo_hybrid", files)
    res["holdout"] = pred_audit("holdout", "exp_E6_holdout_hybrid", files)
    wj(E6 / "final.json", res)
    log(f"final demo S {res['demo']['S']} holdout S {res['holdout']['S']}")


FUNCS = {"truth": st_truth, "sweep": st_sweep, "select": st_select, "final": st_final,
         **{f"m{s}": (lambda s=s: _st_member(s)) for s in SEEDS}}


def run_one(name):
    t0 = time.time()
    status(name, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), pid=os.getpid())
    try:
        FUNCS[name]()
    except BaseException as e:
        status(name, state="failed", error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
        raise
    status(name, state="done", seconds=round(time.time() - t0, 1),
           finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--until", choices=STAGES)
    a = ap.parse_args()
    if a.stage:
        run_one(a.stage)
        return 0
    status("driver", state="running", pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    for name in STAGES:
        if done(name):
            log(f"skip {name} (done)")
        else:
            lf = E6 / "logs" / f"{name}.log.txt"
            lf.parent.mkdir(parents=True, exist_ok=True)
            log(f"start {name} -> {lf}")
            with open(lf, "a") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} start {name}\n")
                fh.flush()
                rc = subprocess.run([PY, "-m", "experiments.driver_e6", "--stage", name], cwd=REPO, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                status("driver", state="failed", failed_stage=name)
                log(f"{name} failed rc={rc}; see {lf}")
                return rc
            log(f"done {name}")
        if a.until == name:
            break
    status("driver", state="done" if not a.until else "paused", finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
