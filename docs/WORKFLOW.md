# End-to-end workflow

One `make` run goes from the FAIR-MAST shot list to scored predictions and a comparison with the
current model. Settings live in `workflow/config.mk`; every output goes under `work/xcheck/`
(git-ignored, override with `WORK=...`). The lab drive is only read (feature diff), never written.

```bash
bash scripts/run_xcheck.sh            # systemd user unit mfi-xcheck, runs `make all`
bash scripts/run_xcheck.sh status     # work/xcheck/STATUS.json
make <target>                         # or run any step by hand; finished steps are skipped
```

Each step leaves a stamp in `work/xcheck/stamps/`; delete a stamp (or `make clean-stamps`) to redo it.
Downloads, labels and feature files are written to a temporary name and renamed, so a killed run
resumes without half-written files. `work/xcheck/STATUS.json` gives the current step, the state of
every step, file counts (parquet, labels, features), the training stages and the free disk.

| target | what it does | main outputs |
|---|---|---|
| `diskcheck` | refuses to start with less than `FREE_GB_MIN` (60 GB) free | |
| `shotlist` | the 3,738 FAIR-MAST shots in `data/fair_mast/mast_pool3.txt` | `shotlist.json` |
| `download` | one parquet row per shot from the FAIR-MAST archive, in chunks of 100 run in parallel | `parquet/` |
| `labels` | EFIT++ scalars and the diverted flag per shot | `labels/` |
| `validation` | the three held-out validation shots 28348, 28350, 28351 and their labels | `validation/` |
| `cache` | 308 input features and the 65x65 flux target per EFIT frame | `cache/` |
| `noplasma` | shots with exactly 200 EFIT slices on the fixed 5 ms clock | `outputs/compat/no_plasma.json` |
| `prep` | seed-42 split, QC, fixed \|I_p\| window, PCA basis, archive validation; holdout list recomputed | `outputs/experiments/common/`, `experiment_splits_seed42.json`, `holdout_seed42.json` |
| `route` | boundary and point labels for the curve-distance objective | `outputs/experiments/common/route.npz` |
| `features` | fresh features vs the lab-drive cache, per shot and column group, with clock offsets and scale ratios | `xcheck/features.json` |
| `smoke` | current model on the freshly built holdout and validation rows; stops the run on failure | `xcheck/smoke.{json,md}` |
| `truth`, `sweep` | truth tables, loss-weight sweep (4 settings x 40 epochs) | `outputs/experiments/E6/` |
| `train` | five seeds 101 ... 505 (`m101` ... `m505`) | `outputs/experiments/E6/m*/` |
| `select` | rank-k ensemble with the gate `SELECT_GATE_S` on `archval100` | `outputs/experiments/E6/select.json` |
| `predict`, `score` | predictions and scores on the validation shots and the sealed holdout | `outputs/audit/exp_E6_*` |
| `compare` / `report` | from-scratch vs current model: holdout S and every term, validation shots, curve distances | `xcheck/compare.{json,md}` |
| `paper` | `paper/MAST.pdf` from the tracked sources | |

Pass rule (`smoke` and `compare`): holdout S no more than `S_DROP_MAX` = 0.01 below the current
model and no score term more than `TERM_DROP_MAX` = 0.10 below it. When the rule fails, the report
lists the feature groups and clocks that differ from the lab-drive rows.

## Data preparation steps

These are the adjustments between the FAIR-MAST archive and the training rows. Each one is a fixed,
deterministic step in the code listed.

| step | rule | where |
|---|---|---|
| EFIT and core Thomson clocks | archive seconds x 1000 to ms | `data/fair_mast/ingest.py` |
| Edge Thomson clock | default `--edge-clock training`: the edge block is kept only when its clock reads as seconds (max < 10) and is converted like the core; other edge clocks are dropped. `--edge-clock overlap` instead picks seconds or ms by overlap with the EFIT window +-50 ms | `data/fair_mast/ingest.py` (`EDGE_CLOCK` in `workflow/config.mk`) |
| Edge Thomson at prediction | the 32 edge columns are set to 0 before the network | `models/m9/predict.py` |
| Edge Thomson in scoring rows | when a row has no edge block, the label file's edge block is copied in (no effect on predictions, which zero it) | `pipeline/datasets.py` |
| Magnetics time base | interpolated onto a fixed grid of 30,000 points from -2000 to 3999.8 ms (0.2 ms), NaN outside the measured range | `data/fair_mast/ingest.py` (`MAG_GRID`) |
| Flux grid | `efm/psirz` is stored on a doubled 129-column R grid; every other column is dropped to give 65x65 on the shot's own `r`, `z` | `data/fair_mast/ingest.py` |
| Inputs on the EFIT clock | currents and Thomson profiles are interpolated to the EFIT times; Thomson is 0 outside its own clock | `models/m9/features.py` |
| Frame window | 9-frame window around each frame (308 -> 924 inputs) | `models/m9/features.py` |
| No-plasma shots | shots with exactly 200 EFIT slices on the fixed 5 ms clock are removed from training | `scripts/scan_no_plasma_shots.py`, `experiments/prep.py` |
| \|I_p\| window | training frames need \|I_p\| >= 328.7293519015368 kA | `experiments/common.py`, `experiments/prep.py` |
| QC | shots whose max \|psi\| is more than 10 MAD above the median are dropped | `models/m9/train.py` (`qc_keep`) |
| Scored frames | only diverted EFIT frames are scored | `data/fair_mast/fetch_efit_labels.py`, `pipeline/datasets.py` |

With `--edge-clock training`, rows built from a fresh download give the same features as the
training rows (`scripts/check_feature_identity.py`).

## Measured schedule

One full run on an Intel Core i9-9900K (16 threads, 62 GB RAM) with an NVIDIA GeForce RTX 5060 Ti
16 GB, started at 13:07:47 and finished at 20:17:13 (7 h 9 min). Times are from `STATUS.json`.

| step | what it does | wall time | bound by |
|---|---|---:|---|
| `download` | 3,738 FAIR-MAST shots, 12 processes | 3 h 37 min | network |
| `cache` | features and flux targets | 1 min 38 s | CPU |
| `noplasma` | no-plasma screen | 33 s | CPU |
| `labels` | EFIT++ scalars, 16 threads | 1 h 46 min | network |
| `prep` | split, QC, \|I_p\| window, PCA, holdout list | 45 s | CPU |
| `route` | boundary and point labels | 7 min 34 s | CPU |
| `features` | diff against the lab-drive cache | 10 s | CPU |
| `validation` | three validation shots | 1 min 41 s | network |
| `smoke` | reported model on rebuilt rows | 1 min 35 s | GPU + CPU |
| `truth` | truth tables | 11 s | GPU |
| `sweep` | 4 settings x 40 epochs | 16 min 40 s | GPU |
| `train` | five seeds (7 min 35 s to 19 min 43 s each) | 1 h 7 min | GPU |
| `select` | rank-k ensembles | 6 min 56 s | GPU + CPU |
| `score` | validation shots and holdout | 1 min 29 s | GPU + CPU |
| `compare` | vs the reported model | < 1 s | CPU |

Disk: 36 GB under `work/xcheck/` (parquet 27 GB, features 6.1 GB, outputs 2.4 GB, labels 90 MB).

Result of that run: all 3,738 rebuilt shots bit-identical to the lab-drive features, clocks with zero
offset and unit scale, same sealed split; the reported model on the rebuilt rows gives holdout
S 0.934444; the retrained ensemble gives holdout S 0.935463 (validation shots 0.898301, reported
model 0.899573).
