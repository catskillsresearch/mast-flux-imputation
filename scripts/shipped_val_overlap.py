"""Reproduce the shipped M9 members' 10% validation shots and their overlap with archval100/select30."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from experiments.common import LAB_CACHE, SHIPPED_IP_LO, SPLITS_JSON
from models.m9 import train as T
from models.m9.features import IP_COL
sealed, pool = T.split(LAB_CACHE)
keep = set(open('data/fair_mast/mast_qc_keep.txt').read().split())
train = [k for k in pool if k in keep]
d = np.load('outputs/experiments/common/data.npz')
dk = [str(k) for k in d['keys']]; cnt = np.bincount(d['ids'], minlength=len(dk))
c = dict(zip(dk, cnt.tolist()))
nop = set(json.load(open('outputs/compat/no_plasma.json'))['no_plasma_keys'])
ids_n = {}
loaded = 0
for i, k in enumerate(train):
    if k in c: n = c[k]
    elif k in nop: n = None
    else: n = None
    if n is None:
        g = T.load_shot(LAB_CACHE, k); loaded += 1
        if g is None: continue
        X, Y = g; m = min(len(X), len(Y))
        n = int((np.abs(X[:m, IP_COL].astype(np.float64)) >= SHIPPED_IP_LO).sum())
    if n > 0: ids_n[i] = n
tot = sum(ids_n.values())
rng = np.random.default_rng(42)
sample_n = min(200_000, tot); n_shots = len(ids_n)
off = 0; picks = []
for sid in sorted(ids_n):
    nn = ids_n[sid]; ix = np.arange(off, off + nn); off += nn
    picks.append(rng.choice(ix, size=min(max(1, sample_n // n_shots), nn), replace=False))
pick = np.concatenate(picks)
if len(pick) > sample_n: pick = rng.choice(pick, size=sample_n, replace=False)
shots = np.array(sorted(ids_n)); rng.shuffle(shots)
nv = max(1, int(0.1 * len(shots)))
val = {train[i] for i in shots[:nv]}
sp = json.load(open(SPLITS_JSON))
out = {"shipped_train_shots_with_frames": n_shots, "shipped_frames": tot, "shipped_val_shots": nv, "loaded_from_cache": loaded}
for s in ("archval100", "select30", "archval"):
    ks = sp[s]; out[s] = {"n": len(ks), "in_shipped_val": sum(k in val for k in ks), "in_shipped_train": sum(k not in val for k in ks)}
print(json.dumps(out, indent=1))
json.dump(out, open('outputs/experiments/common/shipped_overlap.json', 'w'), indent=1)
