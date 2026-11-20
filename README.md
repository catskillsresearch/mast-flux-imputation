# MAST Poloidal Flux Imputation in the Absence of Direct Magnetic Sensors

Reconstruct the poloidal flux $\psi(R,Z)$ of MAST plasmas on a 65 × 65 grid
($R\in[0.06,2.00]$ m, $Z\in[-2,2]$ m), together with $q_{95}$ and $\beta_N$ at every EFIT++ time,
**from coil currents, plasma current, the toroidal-field current and core Thomson scattering
only**. No magnetic probe or flux-loop signal is a model input. (The EFIT++ reconstructions used as
ground truth *were* computed with those sensors; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md).)

The model trains only on open data from the UK FAIR-MAST archive. The scoring metric and the
output template are those of the FEC MAST task. The paper is `paper/MAST.pdf`
(sources: `paper/MAST.md` note tree).

## Model

* **Output space:** a 60-mode PCA basis of FAIR-MAST EFIT++ flux, fitted on the training shots;
  $\hat\psi=\mu+C^{\top}z$.
* **Features:** per EFIT frame, 14 currents ($I_p$ included) and 131 + 131 core-Thomson $T_e$, $n_e$
  channels and 32 edge-Thomson columns (zeroed at prediction), 308 values; a per-shot window-9 moving average and its smoothed
  difference give 924 features, standardised and clipped to ±10.
* **Network:** `CoeffDualContext` — a present-frame residual encoder (308 → 512, 4 blocks) and a
  context encoder (616 → 512, 3 blocks) joined by a zero-initialised gate; a linear head to the 60
  coefficients and five linear scalar heads ($q_{95}$, $\beta_N$, $R_{\rm axis}$, $Z_{\rm axis}$,
  $\ell_i$). The $q_{95}$ and $\beta_N$ heads are the model's scalar outputs.
* **Ensemble:** five seeds (101, 202, 303, 404, 505), averaged.
* **Calibration:** coefficient MSE plus per-shot, range-normalised curve distances of $q_{95}$,
  $\beta_N$ (heads) and $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$, $V$, $R_{\rm axis}$,
  $Z_{\rm axis}$ (differentiable read-outs of $\hat\psi$), weighted towards the end of the shot,
  with smaller boundary, critical-point, temporal and head terms; early stopping and checkpoint
  selection on archive validation shots.

Details: [docs/MODEL.md](docs/MODEL.md) and [docs/TRAINING_REGIME.md](docs/TRAINING_REGIME.md).

## Data

All shots come from FAIR-MAST (`s3://mast/level1/shots/<N>.zarr`, endpoint
`https://s3.echo.stfc.ac.uk`, anonymous, CC BY-SA 4.0): currents (`amc`), Thomson (`ayc`, `aye`)
and EFIT++ (`efm`) as ground truth.

| set | shots | frames | use |
|---|---|---|---|
| training pool after no-plasma removal, QC and the $\lvert I_p\rvert$ window | 3,258 contributing | 211,534 | training (90%) and archive validation (325 shots) |
| sealed holdout (seed 42) | 65 with diverted slices (of 80) | 3,449 | evaluation, once |
| validation shots 28348, 28350, 28351 | 3 | 115 | evaluation |

## Setup and data

```bash
uv venv .venv && uv pip install -e ".[fetch]"     # torch: add --index-url for your CUDA build
python data/fair_mast/ingest.py --pool data/fair_mast/mast_pool3.txt --shots 5000
python data/fair_mast/make_holdout.py             # sealed-80 list + disjointness checks
python data/fair_mast/fetch_efit_labels.py --keys-json data/fair_mast/holdout_seed42.json
# held-out validation shots, straight from FAIR-MAST
python data/fair_mast/ingest.py --keys 28348 28350 28351 --out data/fair_mast/validation/parquet
python data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351 --out data/fair_mast/validation/labels
```

Or point at existing copies with `FAIR_MAST_DIR`, `FAIR_MAST_LABELS_DIR` or a lab root
`CATSKILLS_FUSION_LAB` (see `pipeline/paths.py`). All data, labels, caches, weights and outputs are
git-ignored.

## Reproduce end to end

```bash
bash scripts/run_xcheck.sh            # `make all` as the systemd user unit mfi-xcheck
bash scripts/run_xcheck.sh status     # work/xcheck/STATUS.json: current step, counts, training stages
make smoke                            # or any single step; finished steps are skipped
```

The `Makefile` runs the whole chain from the FAIR-MAST shot list: download, labels, features,
no-plasma scan, split/QC/PCA, a feature diff against an existing data copy, a smoke check of the
current model on the fresh rows, the weight sweep, five seeds, selection, scoring and a comparison
with the current model. Settings are in `workflow/config.mk`; outputs go to `work/xcheck/`
(git-ignored). Steps, the data preparation rules (clocks, grids, edge Thomson, frame windows,
no-plasma removal) and disk and time estimates are in [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Train

```bash
python -m models.m9.train --build-cache   # features, split and PCA basis from FAIR_MAST_DIR
bash scripts/run_e6.sh                    # train the five members, select, predict + audit the holdout
bash scripts/run_e6.sh status             # progress
```

`scripts/run_e6.sh` runs `python -m experiments.driver_e6` as a detached, resumable job; its stages
are `truth`, `sweep`, `m101` … `m505`, `select`, `final`, and any one can be run with
`--stage`. Checkpoints are written under `outputs/experiments/` and are not in git.

## Predict and score

```bash
python scripts/make_compare_preds.py demo holdout   # model predictions + audits -> outputs/compare/<split>_e6.npz
python -m pipeline.score outputs/compare/demo_e6.npz
python -m pipeline.audit outputs/compare/demo_e6.npz --tag compare_demo_e6
MPLCONFIGDIR=/tmp/mpl python scripts/make_model_report.py   # paper tables and figures
```

Predictions are `.npz` files in the template layout: `shot_XXXX_psirz` (T,65,65),
`shot_XXXX_q95` (T,), `shot_XXXX_betaN` (T,), with a `.json` sidecar giving the shot order. Models
only see `inputs_only(row)`, which removes every `efit_*` label column.

## Metric

$$
S = 0.55\,\max(0,R^2_\psi) + 0.15\,\max\!\Big(0,\tfrac{R^2_{q_{95}}+R^2_{\beta_N}}{2}\Big)
 + 0.10\,\big(1-\min(1,D_{\rm LCFS})\big) + 0.20\cdot\tfrac17\sum_{j=1}^{7}\max(0,R^2_{{\rm cons},j})
$$

$R^2$ is pooled over all frames (and pixels) of a fold; $D_{\rm LCFS}$ is the per-frame Hausdorff
distance between the LCFS contours over $R_{\rm geo}$, clipped at 1; the seven consistency scalars
($R_{\rm axis}$, $Z_{\rm axis}$, $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$, $V$, $\ell_i$) are
derived by the same code from predicted and EFIT++ $\psi$. The FEC scoring modules (v3.1.0) are
vendored byte-identical in `fec_scoring/`; `pipeline/score.py` ports the fold reduction.

## Results

Ground truth (FAIR-MAST EFIT++) against the model's prediction, from
`paper/generated/model_scorebox.md` (written by `scripts/make_model_report.py` from the saved
prediction and audit files):

| | 28348 | 28350 | 28351 | 3 validation shots | sealed holdout (65 shots) |
|---|---|---|---|---|---|
| $S$ | 0.874374 | 0.910541 | 0.905806 | 0.899573 | 0.934444 |
| $R^2_\psi$ | 0.998652 | 0.998881 | 0.999105 | 0.998879 | 0.997832 |
| $R^2_{q_{95}}$ | 0.964105 | 0.984437 | 0.977610 | 0.976192 | 0.913296 |
| $R^2_{\beta_N}$ | 0.533545 | 0.701931 | 0.734513 | 0.663238 | 0.677889 |
| $1-D_{\rm LCFS}$ | 0.951246 | 0.960200 | 0.947812 | 0.953086 | 0.938632 |
| Consistency | 0.588334 | 0.693293 | 0.665541 | 0.659619 | 0.862171 |

Curve distances, boundary distances, time series and overlays are in the Results section of the
paper and, shot by shot, in the viewer.

## Viewer

```bash
python -m app.fec_site --port 8770      # http://127.0.0.1:8770/
```

Per split, fold scorecards and a per-shot table; per shot, EFIT++ against model predictions: $\psi$ contours
with LCFS, O-point and X-points, residual maps, $q_{95}$/$\beta_N$/shape traces, a time scrubber and
per-frame scorecards.

## Build the paper

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/make_model_report.py
bash scripts/build_paper_pdf.sh         # paper/MAST.md -> paper/arxiv.md -> paper/arxiv.tex -> paper/MAST.pdf
```

Needs `pip install -e .[paper]`, pandoc, latexmk/pdflatex and Graphviz
(`dot -Tpdf paper/figures/src/model_arch.dot` re-renders the diagram). The build fails on any
undefined citation or reference. Tables in `paper/generated/` are written from saved outputs, so the
paper never hand-copies a number.

## Repository layout

```
data/fair_mast/      ingest.py, make_holdout.py, fetch_efit_labels.py, holdout_seed42.json, QC manifests
fec_scoring/         FEC scoring modules (byte-identical) + mast_envelope.npz
models/m9/           features.py, network.py, train.py (data cache, PCA basis)
experiments/         e6.py (loss, read-outs, trainer), driver_e6.py, evaluate.py
pipeline/            paths.py, datasets.py, score.py, audit.py
app/fec_site/        MAST viewer (stdlib HTTP server + Plotly)
scripts/             run_e6.sh, make_compare_preds.py, make_model_report.py, paper build
paper/               note tree, generated tables, figures, MAST.pdf
tests/               label test and scorer cross-check scripts
```

Other modules and scripts in the tree are not part of the documented workflow.

Licence: Apache License 2.0 (see `LICENSE` and `NOTICE`); Copyright 2026 Lars Warren Ericson (d/b/a
Catskills Research Company). Data licences: FAIR-MAST CC BY-SA 4.0; FEC scoring code and template
CC BY 4.0.
