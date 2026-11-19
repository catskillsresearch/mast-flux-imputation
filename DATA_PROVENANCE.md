# Data provenance

What every model input, label and parameter in this repository comes from, and what the model was
trained, selected and evaluated on.

## 1. Sources

| source | location | licence | contents used |
|---|---|---|---|
| FAIR-MAST | `s3://mast/level1/shots/<N>.zarr`, endpoint `https://s3.echo.stfc.ac.uk`, anonymous | CC BY-SA 4.0 | `efm` (EFIT++ ground truth), `amc` (currents), `ayc` (core Thomson), `aye` (edge Thomson) for every training, validation and holdout shot |
| FEC scoring code v3.1.0 | `github.com/Sophelio/fusion-equilibrium-challenge-starter`, `scoring/` | per repository | `fec_scoring/*.py` and `masks/mast_envelope.npz`, byte-identical; the metric and the output template |
| catskills_fec | local repository | proprietary (not copied; this repository is Apache-2.0) | model, ingest and viewer code, ported |

## 2. Model inputs

Coil currents (EFPS, SOL, TF, P2–P6 upper/lower), plasma current $I_p$ and core Thomson $T_e$,
$n_e$ (131 channels each), from FAIR-MAST `amc` and `ayc`. Edge-Thomson columns are present in the
rows but zeroed by the model. **No magnetic probe, flux loop or EFIT-derived quantity is a model
input.** The EFIT++ ground truth is itself a magnetics-constrained reconstruction, so the labels
encode the probes; only the model is probe-free. `pipeline/datasets.inputs_only()` strips every
`efit_*` column except `efit_times` and the grid before the model sees a row.

## 3. Labels

* $\psi$: EFIT++ `efm/psirz` on the EFIT++ grid `efm/r`, `efm/z`.
* $q_{95}$, $\beta_N$, $R_{\rm axis}$, $Z_{\rm axis}$, $\ell_i$: `efm` scalars, read by
  `data/fair_mast/fetch_efit_labels.py`.
* Diverted (scored) frames: any finite `efm/xpoint{1,2}_rc`; 36, 42 and 37 frames on the
  validation shots 28348, 28350 and 28351.

## 4. What the model was fitted and selected on

1. Training targets, PCA basis, mean and feature statistics: FAIR-MAST EFIT++ of the training
   shots (3,258 contributing shots, 211,534 frames, minus the 325 archive-validation shots).
2. The 335 no-plasma shots of the pool are removed before quality control; the $|I_p|$ window is
   fixed at 328.7293519 kA.
3. Loss weights, early stopping and checkpoint selection: archive validation (`archval100` for the
   objective $J$, all 325 shots for the selection score).
4. The soft read-outs use the vessel-envelope mask of the scoring code where no EFIT++ LCFS exists.

## 5. What the model was evaluated on

* **Sealed holdout:** `data/fair_mast/holdout_seed42.json`, 80 FAIR-MAST shots chosen by
  `random.Random(42)` from the ingested keys minus the validation shots; 65 have diverted slices
  (3,449 frames). `data/fair_mast/make_holdout.py` checks that it is disjoint from the training
  pool and the validation shots.
* **Validation shots 28348, 28350, 28351:** downloaded from FAIR-MAST by
  `data/fair_mast/ingest.py --keys …` and `fetch_efit_labels.py --shots …` into
  `data/fair_mast/validation/` (git-ignored), never used in training or selection.

## 6. Static tables in the archive rows

The archive training rows hold coil-geometry and Thomson-chord tables in the template layout,
filled from the template's example row. The model reads none of these columns; the grid it uses
equals `efm/r`, `efm/z`. `data/fair_mast/ingest.py` in this repository fills the grid from `efm`
and writes no other static table.

## 7. Not in git

Raw parquet rows, labels, caches, trained `.pt` weights and all outputs are git-ignored; only code,
manifests (`holdout_seed42.json`, QC lists), `fec_scoring/masks/mast_envelope.npz`, the paper
sources and the generated tables are tracked.
