# Data provenance

What every model input, label and parameter in this repository comes from, and what each
component was trained, tuned or evaluated on. Every statement below was checked against code or
data (2026-11-18).

## 1. Sources

| source | location | licence | contents used |
|---|---|---|---|
| FEC public data | Hugging Face `Sophelio/fusion-equilibrium-challenge`, `data/mast_public_test/` | CC BY 4.0 | 1,206 MAST input rows, no labels |
| FEC starter kit | `github.com/Sophelio/fusion-equilibrium-challenge-starter`, `parquet_data/` (Git LFS) | per repository | 3 labelled MAST rows (28348, 28350, 28351), checksums in `data/fec/fetch.py` |
| FEC scorer v3.1.0 | starter kit `scoring/` | per repository | `fec_scoring/*.py` and `masks/mast_envelope.npz`, byte-identical |
| FAIR-MAST | `s3://mast/level1/shots/<N>.zarr`, endpoint `https://s3.echo.stfc.ac.uk`, anonymous | CC BY-SA 4.0 | groups `efm` (EFIT++), `amc` (currents), `ayc` (core Thomson), `aye` (edge Thomson) |
| catskills_fec | local repository, HEAD `a8f66617` | proprietary (catskills_fec `LICENSE`, not copied; this repo is Apache-2.0) | OSF code, M9 code, ingest, viewer (ported) |

FEC (rows) and FAIR-MAST (archive) carry **bit-identical EFIT++ $\psi$** for the same shot and
time; FEC keeps only diverted slices, rounds `efit_times` to whole ms, and resamples the currents
differently (see §5).

## 2. Model inputs

Coil currents (EFPS, SOL, TF, P2–P6 upper/lower), plasma current $I_p$, core Thomson $T_e,n_e$
(131 channels) and, for OSF's $\beta_N$, Thomson inside the LCFS. Static tables (coil geometry,
Thomson chord radii) are inputs supplied by the challenge. **No magnetic probe, flux loop, or
EFIT-derived quantity is a model input.** Note that the EFIT++ ground truth itself is a
magnetics-constrained reconstruction, so the labels encode the probes; only the models are
probe-free. `pipeline/datasets.inputs_only()` strips every `efit_*` column except `efit_times` and
the grid before a model sees a row.

## 3. Labels

* $\psi$: EFIT++ `efm/psirz` (FAIR-MAST) = FEC `efit_psirz` (verified bit-identical).
* $q_{95}$, $\beta_N$: `efm/q_95`, `efm/betan` = FEC `efit_q95`, `efit_beta_n` (verified exactly on
  the 3 demo shots, `tests/test_labels.py`; $q_{95}$ is positive in both, no `abs`).
* Diverted frame rule: any finite `efm/xpoint{1,2}_rc` (reproduces the FEC frame count 36/42/37).
* The lab directory `datasets/.../data/mast_labeled` (1,207 files) holds **synthetic Picard
  labels**, not EFIT; it is not used anywhere here.

## 4. Parameters

### OSF (`models/osf/calibration.py`)
38 `CALIBRATED` scalars, set by inspecting physics end-members on the 3 labelled starter-kit shots
plus a small number of aggregate scores on `mast_public_test`. There is no fitting code for them;
the values are copied exactly. The hybrid's $q_{95}$/$\beta_N$ read-out maps (E5:
$a_q=0.626$, $b_q=1.957$, $a_b=1.013$, $b_b=0.093$) are fitted by Nelder–Mead to FAIR-MAST EFIT++
labels on 200 training shots (`experiments/osf_recal.py`).

### M9 (`models/m9`, `experiments/`)
Targets and PCA basis: FAIR-MAST EFIT++ `psirz` of the training pool (3,738 ingested − 3 demo −
80 sealed = 3,655; minus 335 no-plasma shots; QC keeps 3,280). With the $|I_p|$ window fixed at
328.7293519 kA, 3,258 shots contribute 211,534 frames; 10 % of those shots (seed 42, 325 shots)
are the archive validation split used for checkpoint selection. Checkpoints are written under
`outputs/experiments/` and are not in git.

## 5. Holdout (sealed-80) and its disjointness

Rule (`data/fair_mast/make_holdout.py`, manifest `holdout_seed42.json`): `sorted(ingest keys) −
{demo 3}`, `random.Random(42).shuffle`, first 80. This is the split the M9 training code uses (seed 42). Recomputed checks, all true:

| check | value |
|---|---|
| ingested shots | 3,738 |
| recomputed training pool == QC "train" set (keep ∪ quarantine) | true (3,655) |
| keep ∩ sealed, pool ∩ sealed, demo ∩ sealed | 0, 0, 0 |
| quarantine-sealed ⊆ sealed | true (`mast_shot_27373`) |

Evaluation cut: 15 sealed shots have no diverted slice (24901, 26506, 29932, 24876, 28295, 27681,
23248, 27621, 27720, 29297, 26273, 27819, 27778, 23424, 30442) → **65 shots, 3,449 frames**. 27373
is kept (it is a holdout shot, quarantined only from training).

Holdout processing vs. real FEC rows (limitations):
* currents come from the ingest's resampling of `amc`, which differs from the FEC rows (e.g.
  max $|\Delta I_p|$ = 44 kA on 28348; magnetics clock offset 0.1 ms);
* the lab ingest predates an edge-clock fix, so edge Thomson is overlaid from `aye` (60 of 65
  shots); the chord radius is the shot's own `aye` mean (FEC uses campaign tables);
* core `thomson_core_R` keeps the ingest's 131-entry archive radius (FEC rows: 130 chord radii).

## 6. Data use and tuning facts

1. M9 targets and PCA basis are FAIR-MAST EFIT++ `psirz` from the QC-kept archive pool. This pool
   very likely overlaps the FEC hidden test folds; M9 is not a zero-shot result.
2. The FAIR-MAST pool (≈4,566 shots with EFIT + Thomson, of which FEC curated 2,416) very likely
   overlaps the FEC hidden folds.
3. The 335 no-plasma shots of the pool (reduced `efm` product: 200 fixed 5 ms slices, no scalars
   or X-points, max|Ip| ≤ 46.8 kA) are removed before QC. They would set the 2-means window
   threshold (328.73 kA with them, 471.99 kA without), so the window is a fixed constant. 14 of
   the sealed 80 are such shots.
4. The base loss is MSE on the 60 PCA coefficients; the route-guided experiments add boundary,
   critical-point, temporal, scalar-head and (E6) curve-distance terms (`docs/TRAINING_REGIME.md`).
5. M9 trains on all $|I_p|$-windowed frames, limited ones included.
6. OSF `_extract_lcfs` loads `mast_envelope.npz` `mask_coarse` (the scorer's LCFS extractor) for the
   $q_{95}$/$\beta_N$ read-outs.
7. OSF shape-parameter choices were made with reference to $R^2$ values on the labelled demo shots
   and a small number of aggregate public-test scores.
8. OSF's $q_{95}$ uses the vacuum $F_b$ and an empirical factor law.
9. OSF is a Picard iteration of up to 20 passes ("one-shot" = no fit to measured flux).
10. Ingest and FEC demo rows carry bit-identical $\psi$; they differ in frame population (all slices
    vs diverted).
11. No magnetic sensor is a model input; the EFIT++ ground truth used magnetic sensors.
12. Holdout numbers are label-disjoint on the sealed-80 for every model and read-out map.
13. Holdout input processing differs from FEC rows (§5).
14. This repository is Apache-2.0; copyright is retained by Lars Warren Ericson (d/b/a Catskills
    Research Company).

## 7. What is *not* in git

All parquet/npz data (except the scorer's `mast_envelope.npz`), FAIR-MAST labels, M9 checkpoints
(`*.pt`), caches and `outputs/`. No DIII-D data or models are included.
