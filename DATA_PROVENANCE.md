# Data provenance

What every model input, label and parameter in this repository comes from, and where the
statements in the source repository (catskills_fec `PROVENANCE.md`, `docs/mast-calibration.md`,
script docstrings) do **not** match its code. Every statement below was checked against code or
data during the extraction (2026-11-18).

## 1. Sources

| source | location | licence | contents used |
|---|---|---|---|
| FEC public data | Hugging Face `Sophelio/fusion-equilibrium-challenge`, `data/mast_public_test/` | CC BY 4.0 | 1,206 MAST input rows, no labels |
| FEC starter kit | `github.com/Sophelio/fusion-equilibrium-challenge-starter`, `parquet_data/` (Git LFS) | per repository | 3 labelled MAST rows (28348, 28350, 28351), sha256 in `data/fec/fetch.py` |
| FEC scorer v3.1.0 | starter kit `scoring/` | per repository | `fec_scoring/*.py` and `masks/mast_envelope.npz`, byte-identical |
| FAIR-MAST | `s3://mast/level1/shots/<N>.zarr`, endpoint `https://s3.echo.stfc.ac.uk`, anonymous | CC BY-SA 4.0 | groups `efm` (EFIT++), `amc` (currents), `ayc` (core Thomson), `aye` (edge Thomson) |
| catskills_fec | local repository, HEAD `a8f66617` | proprietary (catskills_fec `LICENSE`, not copied; this repo is Apache-2.0) | OSF code, M9 code + checkpoints, ingest, QC lists, viewer |

FEC (rows) and FAIR-MAST (archive) carry **bit-identical EFIT++ $\psi$** for the same shot and
time; FEC keeps only diverted slices, rounds `efit_times` to whole ms, and resamples the currents
differently (see §5).

## 2. Model inputs (both models)

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
38 `CALIBRATED` scalars. Per catskills_fec `docs/mast-calibration.md`, they were set by inspecting
physics end-members on the 3 labelled starter-kit shots plus 5 Codabench aggregate-score probes on
`mast_public_test`. There is no fitting code; the values are copied exactly. The post-hoc hybrid
factors $c^\*=1.062242$ (with $q_{\rm ref}=1.20$) and $s^\*=1.078508$ were obtained from Codabench
leaderboard probes on the public-test fold.

### M9 (`models/m9`)
Targets and PCA basis: FAIR-MAST EFIT++ `psirz` of the 3,622 QC-kept archive shots (training
pool = 3,738 ingested − 3 demo − 80 sealed = 3,655; QC removes 33). After the $|I_p|\ge
328.7293519015368$ kA window (2-means on $|I_p|$, seed 42), 3,265 shots contribute frames; 10 % of
those shots (seed 42) are the validation split used for early-stopping selection (best epochs
46/49/46). Shipped checkpoints: `archive/artifacts/CRC_FEQ_mast_M9_win50_i{101,202,303}.pt`, sha256
`7660c1d3…`, `06ba7ef2…`, `de3b6950…` (match `SHA256SUMS`; full hashes in `models/m9/predict.py`).

## 5. Holdout (sealed-80) and its disjointness

Rule (`data/fair_mast/make_holdout.py`, manifest `holdout_seed42.json`): `sorted(ingest keys) −
{demo 3}`, `random.Random(42).shuffle`, first 80. This is the split the M9 training code uses
(`seed=42` stored in every checkpoint). Recomputed checks, all true:

| check | value |
|---|---|
| ingested shots | 3,738 |
| recomputed training pool == QC "train" set (keep ∪ quarantine) | true (3,655) |
| QC keep list == `mast_qc_keep.txt` | true (3,622) |
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

## 6. Discrepancies between catskills_fec documentation and its code

| # | documentation says | code / data shows |
|---|---|---|
| 1 | `PROVENANCE.md`: "FAIR-MAST EFIT++ / archive reconstructions: No"; "M9 trained without forbidden MAST equilibrium labels" | M9 targets and PCA basis are FAIR-MAST EFIT++ `psirz` from 3,622 archive shots. The training code's own note: "overlaps hidden test fold; not a zero-shot claim". |
| 2 | "`mast_envelope.npz` as prior/mask: No" | OSF `_extract_lcfs` loads `mast_envelope.npz` `mask_coarse` (the scorer's LCFS extractor) for the $q_{95}$/$\beta_N$ read-outs. |
| 3 | "Labelled MAST demo equilibria as fit targets: No"; "zero fitting against 3 demo shots" | `mast-calibration.md` justifies shape-parameter choices with $R^2$ values that require labelled MAST frames; the M9 run's PASS gate is demo $R^2_\psi\ge0.9976$. |
| 4 | $c^\*$, $s^\*$ are "frozen code hyperparameters" | fitted from Codabench leaderboard probes on the public-test fold. |
| 5 | `generate_final_predictions.py` docstring: $s^\*=1.000000$ | code uses 1.078508. |
| 6 | "paramagnetic $c=1.20$ baseline" | not in the solver: $q_{95}$ uses vacuum $F_b$ and an empirical factor law; the final map multiplies by $c^\*/1.20$. |
| 7 | "one-shot" OSF | Picard iteration, up to 20 passes. |
| 8 | ingest and demo differ by "recording / $\psi$ scale" | $\psi$ is bit-identical; the difference is the frame population (all slices vs diverted). |
| 9 | checkpoint metadata `train_shots = 3622` | 3,265 contribute frames after the $|I_p|$ window; 10 % of shots are validation. |
| 10 | — (clarification) | loss is MSE on the 60 PCA coefficients only; scalar heads untrained. |
| 11 | — (clarification) | trained on all $|I_p|$-windowed frames, limited ones included. |
| 12 | — (clarification) | shipped weights are in `archive/artifacts` (match `SHA256SUMS`); `artifacts/` holds an Oct 30 rebuild with different hashes, and the P36 result JSONs there were overwritten. |
| 13 | "no magnetic sensors" | true for model inputs; the EFIT++ GT used them. |
| 14 | OSF holdout numbers are blind | label-disjoint on the sealed-80, but $c^\*,s^\*$ and shape choices were tuned via public-test probes and the demo shots. |
| 15 | archive ⟂ FEC folds | the FAIR-MAST pool (≈4,566 shots with EFIT + Thomson, of which FEC curated 2,416) very likely overlaps the FEC hidden folds. |
| 16 | — | The proprietary catskills_fec `LICENSE` was initially copied here; resolved by the owner: this repository is Apache-2.0, copyright retained by Lars Warren Ericson (d/b/a Catskills Research Company). |
| 17 | — | holdout input processing differs from FEC rows (§5). |

## 7. What is *not* in git

All parquet/npz data (except the scorer's `mast_envelope.npz`), FAIR-MAST labels, M9 checkpoints
(`*.pt`), caches and `outputs/`. No DIII-D data, models or competition-only artefacts (O1G,
brown, TF_LS_O1, Weddy_dyn40, phase-3 rebuilds, bundles, Codabench rules) were copied.
