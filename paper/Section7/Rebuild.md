# Rebuild from scratch

`bash scripts/run_xcheck.sh` runs `make all`: every step from the FAIR-MAST shot list to the scored
ensemble, with nothing taken from an existing data copy. The table gives the measured schedule of one
such run on a single workstation (Intel Core i9-9900K, 8 cores / 16 threads, 62 GB RAM, NVIDIA
GeForce RTX 5060 Ti 16 GB). Steps run one after another; the download runs 12 processes on chunks
of 100 shots and the label fetch 16 threads.

| step | what it does | wall time | bound by |
|---|---|---:|---|
| `download` | 3,738 FAIR-MAST shots, one parquet row each | 3 h 37 min | network |
| `cache` | 308 inputs and the $65\times65$ flux target per frame | 1 min 38 s | CPU |
| `noplasma` | no-plasma screen | 33 s | CPU |
| `labels` | EFIT++ scalars and diverted flag, 3,738 shots | 1 h 46 min | network |
| `prep` | split, QC, $\lvert I_p\rvert$ window, PCA basis, holdout list | 45 s | CPU |
| `route` | boundary and point labels, 3,258 shots | 7 min 34 s | CPU |
| `features` | feature diff against an existing copy | 10 s | CPU |
| `validation` | the three validation shots and their labels | 1 min 41 s | network |
| `smoke` | reported model on the rebuilt holdout and validation rows | 1 min 35 s | GPU + CPU |
| `truth` | truth tables | 11 s | GPU |
| `sweep` | 4 loss-weight settings $\times$ 40 epochs | 16 min 40 s | GPU |
| `train` | five seeds (7 min 35 s to 19 min 43 s each) | 1 h 7 min | GPU |
| `select` | rank-$k$ ensembles on `archval100` | 6 min 56 s | GPU + CPU |
| `score` | validation shots and sealed holdout | 1 min 29 s | GPU + CPU |
| `compare` | comparison with the reported model | < 1 s | CPU |
| **total** | | **7 h 9 min** | |

Table: Measured schedule of a full rebuild from FAIR-MAST.

The download and the label fetch take 5 h 23 min of the 7 h 9 min; the steps that use the GPU take 1 h 34 min. The
run uses 36 GB of disk: parquet rows 27 GB, feature files 6.1 GB, training data, checkpoints and
predictions 2.4 GB, labels 90 MB.

Rebuilding from FAIR-MAST alone gives the same inputs bit for bit: for all 3,738 shots the currents,
core and edge Thomson columns and the flux targets are identical to the rows the reported model was
trained on, every time axis agrees with zero offset and unit scale, and the sealed split is the same.
The reported model scores holdout $S$ = 0.934444 on the rebuilt rows. The ensemble retrained from
scratch scores holdout $S$ = 0.935463 and 0.898301 on the three validation shots, against 0.934444
and 0.899573 for the reported model; the difference is seed-level and GPU non-determinism. The model
and numbers reported in this paper are those of the reported ensemble.
