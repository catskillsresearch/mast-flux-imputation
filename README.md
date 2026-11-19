# MAST Poloidal Flux Imputation in the Absence of Direct Magnetic Sensors

Reconstruct the poloidal flux $\psi(R,Z)$ of MAST plasmas, together with $q_{95}$ and $\beta_N$, on
the 65 × 65 Fusion Equilibrium Challenge (FEC) grid ($R\in[0.06,2.00]$ m, $Z\in[-2,2]$ m), **from
coil currents, plasma current, the toroidal-field current and Thomson scattering only**. No magnetic
probe ($B_\theta$ pick-up coil) or flux-loop signal is an input to either model. (The EFIT++
reconstructions used as ground truth *were* computed with those sensors; see
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).)

Two models are provided:

| model | $\psi(R,Z)$ | $q_{95}$, $\beta_N$ | data-driven? |
|---|---|---|---|
| **OSF** (`mast_12_osf_0.0.12`) | free-boundary Grad–Shafranov forward solve | flux-surface read-out of the OSF map + Thomson pressure | 38 hand-calibrated physics parameters |
| **M9 + OSF** (hybrid, shipped Phase-2 Final recipe) | 3-member M9 ensemble (PCA-60 + dual-context residual MLP, trained on FAIR-MAST EFIT++) | OSF read-outs × post-hoc factors | yes (M9 weights) |

Everything here was extracted from the Catskills Research FEC competition repository and re-verified
bit-for-bit (see [Verification](#9-verification)). Mathematical detail: [docs/MODEL.md](docs/MODEL.md).

---

## 1. Model description

### 1.1 Grad–Shafranov forward problem (OSF)

Axisymmetric ideal-MHD equilibrium in $(R,\phi,Z)$ with poloidal flux $\psi$ (Wb rad$^{-1}$, FEC
convention) satisfies

$$
\Delta^{*}\psi \;\equiv\; R\,\frac{\partial}{\partial R}\!\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right)+\frac{\partial^{2}\psi}{\partial Z^{2}}
\;=\; -\mu_0 R\, j_\phi ,\qquad
j_\phi = R\,p'(\psi)+\frac{F F'(\psi)}{\mu_0 R}.
$$

On MAST the stored EFIT++ flux has its magnetic axis at a **maximum** of $\psi$:
$\sigma_{\rm ax}=+1$ (DIII-D is $-1$). OSF solves the free-boundary problem with the Green's
function of $\Delta^*$ (no vessel/eddy currents):

$$
\psi(\mathbf x)=\underbrace{\sum_{c} \lambda_c(w,b)\, I_c \sum_{e\in c}\frac{G(\mathbf x,\mathbf x_e)}{n_c}}_{\psi_{\rm vac}}
\;+\;\int_{\Omega_p} G(\mathbf x,\mathbf x')\, j_\phi(\mathbf x')\, dA',
\qquad
G=\frac{\mu_0\sqrt{RR'}}{2\pi k}\Big[(2-k^2)K(k)-2E(k)\Big],
$$

$k^2 = 4RR'/[(R+R')^2+(Z-Z')^2]$. $\psi_{\rm vac}$ uses the FEC MAST coil table (812 rectangles:
656 solenoid rows, each one turn carrying $I_{\rm sol}$, and 156 P2–P6 elements, each sharing its
circuit's current; every rectangle is a 3 × 3 sub-filament mean). TF and EFPS do not enter
$\psi_{\rm vac}$; $\lambda_c$ depends on $b$ only.

Two "clocks" are computed from the measured currents at each EFIT time:

$$
w=\Big(1-\min\Big(\tfrac{(|I_{P2U}|+|I_{P2L}|)/2}{|I_p|\,r_\star},1\Big)\Big)^2,\qquad
b=\mathrm{clip}\big(I_{\rm sol}/I_\star,0,1\big),\qquad r_\star=0.34,\ I_\star=16.3\ {\rm kA},
$$

($w$: divertor-coil "walk", $b$: solenoid "bleed"). They set an elliptical current support
$\Omega_p$ (centre $R_0=0.82+0.10w-0.02b$, $Z_0=0.009-0.004b$; half-axes $a_R=0.50+0.08w$,
$a_Z=a_R+0.08+0.42b+0.18(1-b)$), coil multipliers $\lambda_c$ and the enclosed-current factor
$k=0.70+0.30w+0.15b+0.20(1-b)$. On $\Omega_p$, with $\tau=\rho^2$ the geometric normalised radius,

$$
p'\propto(1-\tau),\qquad FF'=-\bar\gamma\,(1-\tau),\ \bar\gamma=-1,\qquad \int_{\Omega_p} j_\phi\,dA = k\,I_p ,
$$

and the $p'$ amplitude is $\langle p\rangle/(\Delta\psi\,\langle s\rangle_V)$ with
$\langle p\rangle=\beta_N^{(0)} B_t I_p/(200\,\mu_0 a)$ from a first-guess $\beta_N^{(0)}$. Because
$\Delta\psi=\psi_{\rm ax}-\psi_b$ depends on the solution, the solve is a Picard iteration (≤ 20 passes,
stop at $\max|\delta\psi|<10^{-4}$ and relative change $<10^{-3}$) — "one-shot" refers to the absence
of any fit to measured flux, not to a single pass.

**Read-outs.** $q_{\rm map}=F_b W/2\pi$ with $W=\oint d\ell/(R|\nabla\psi|)$ on the 95 % surface and
the *vacuum* $F_b=\mu_0N_{\rm TF}I_{\rm TF}/2\pi$ ($N_{\rm TF}=24$), followed by an empirical
well/walk/hole factor law ($q_{\rm hi}=17,\,q_{\rm lo}=8,\,\alpha_q=0.55,\,\eta_q=0.35,\,\zeta_q=2,\,p_q=3$).
$\beta_N = 100\,\langle p\rangle\,2\mu_0 a/(B_t I_p[{\rm MA}])$ with $\langle p\rangle$ the mean of
$2n_eT_e e$ over Thomson chords inside the extracted LCFS; then clip $[0.2,4]$, 7-frame median,
$\times(1-0.2w)$.

### 1.2 M9 (archive-trained flux predictor)

$$
\hat\psi = \frac13\sum_{s\in\{101,202,303\}} \Big(\mu+C^{\!\top} f_s\big(\mathrm{clip}_{\pm10}((A(x)-m)/\sigma)\big)\Big),
\qquad C\in\mathbb R^{60\times4225}\ (\text{PCA of EFIT++ }\psi).
$$

$x_t\in\mathbb R^{308}$: 13 coil currents + $I_p$ interpolated to the EFIT time (14), 131 core-Thomson
$T_e$ and 131 $n_e$ channels interpolated in time (262), and 16 + 16 edge-Thomson channels (32,
zeroed at inference by the shipped recipe). $A$ appends a
centred 9-frame moving average and its smoothed central difference ($\to924$ features). $f_s$ is a
dual-context residual MLP (present-frame stem 308→512 + 4 residual blocks; context stem 616→512 + 3
blocks, zero-initialised gate). Trained with MSE on PCA coefficients on 3,622 QC-passed FAIR-MAST
shots (frames with $|I_p|\ge 328.73$ kA). Details: [models/m9/README.md](models/m9/README.md).

### 1.3 Hybrid (shipped recipe)

$\psi$ = M9 ensemble mean; $q_{95}=\mathrm{clip}(q_{95}^{\rm OSF}\cdot1.062242/1.20,\,2,\,25)$,
$\beta_N=\mathrm{clip}(\beta_N^{\rm OSF}\cdot1.078508,\,0.2,\,5)$. The two factors were obtained from
Codabench leaderboard probes on the FEC public-test fold (see DATA_PROVENANCE.md).

---

## 2. Data description

| set | source | shots | labels | use here |
|---|---|---|---|---|
| FEC `mast_public_test` | HF `Sophelio/fusion-equilibrium-challenge` (CC BY 4.0) | 1,206 | none (withheld) | inference / viewer only |
| FEC labelled demo | starter kit `parquet_data/` (Git LFS) | 3 (28348, 28350, 28351) | $\psi$, $q_{95}$, $\beta_N$, … | sanity split |
| FAIR-MAST archive | `s3://mast/level1/shots/<N>.zarr` at `s3.echo.stfc.ac.uk` (CC BY-SA 4.0) | 3,738 ingested | EFIT++ `efm` | M9 training + **sealed holdout** |

One parquet row per shot. Inputs: `magnetics_*_current` (kA; 13 coils + plasma) on a 0.2 ms grid
(30,000 samples, −2000…4000 ms), core Thomson `thomson_core_{times,Te,ne,R}` (~2 ms cadence,
131 channels; 130 radii in the FEC rows), edge Thomson (16 channels, $R\approx1.33$–1.49 m), chord table (146 chords, $Z=0$), coil table (812 rows), `efit_times` and the
grid. Labels: `efit_psirz` (T × 65 × 65), `efit_q95`, `efit_beta_n`. FEC keeps **diverted** EFIT
slices only; the archive keeps all slices.

---

## 3. Acquiring the data

```bash
uv venv .venv && uv pip install -e ".[fetch]"     # torch: add --index-url for your CUDA build
python data/fec/fetch.py --demo                   # 3 labelled demo shots (sha256-checked)
python data/fec/fetch.py --public-test            # 1,206 public-test shots (~GB)
python data/fair_mast/ingest.py --pool data/fair_mast/mast_pool3.txt --shots 5000   # archive -> FEC schema
python data/fair_mast/make_holdout.py             # sealed-80 list + disjointness proof
python data/fair_mast/fetch_efit_labels.py --keys-json data/fair_mast/holdout_seed42.json
```

Or point at existing copies: `FEC_MAST_DIR`, `FEC_DEMO_DIR`, `FAIR_MAST_DIR`, `FAIR_MAST_LABELS_DIR`,
`M9_WEIGHTS_DIR`, or a lab root `CATSKILLS_FUSION_LAB` (see `pipeline/paths.py`). All data,
labels, caches, weights and outputs are git-ignored. The M9 checkpoints (3 × ~10 MB, sha256 in
`models/m9/predict.py`) are not in git; place them in `models/m9/weights/`.

## 4. Calibrating

* **OSF** has no training step. Its parameters live in `models/osf/calibration.py`, split into
  `MACHINE_SPEC` (hardware/schema), `GRID_GATES` (search boxes), `SWITCHES` (code path) and the 38
  `CALIBRATED` physics parameters, plus `FINAL_SCALARS` (the two post-hoc hybrid factors).
* **M9** re-training (recipe identical to the shipped members; reproduces the 3,622-shot QC keep
  list and $I_p^{\rm lo}=328.7293519015368$ kA exactly):

```bash
python -m models.m9.train --build-cache          # features from FAIR_MAST_DIR -> data/fair_mast/cache
python -m models.m9.train --member all           # 3 x 50 epochs -> models/m9/weights/
```

## 5. Applying (CLI)

```bash
python -m pipeline.predict --split holdout --mode osf       # outputs/holdout_osf.npz (+ .json)
python -m pipeline.predict --split holdout --mode hybrid    # outputs/holdout_hybrid.npz
python -m pipeline.predict --split public_test --mode hybrid --workers 14
python -m pipeline.score outputs/holdout_hybrid.npz         # table + outputs/holdout_hybrid.score.json
```

`--split {demo,holdout,public_test}`. Output npz keys follow the FEC submission format:
`shot_XXXX_psirz` (T,65,65), `shot_XXXX_q95` (T,), `shot_XXXX_betaN` (T,), XXXX = index in the
split order recorded in the `.json` sidecar. Models only ever see `inputs_only(row)` (all `efit_*`
label columns removed).

## 6. Visualising

```bash
python -m app.fec_site --port 8770      # http://127.0.0.1:8770/
```

MAST-only: per split, fold scorecards and a per-shot table; per shot, EFIT++ vs OSF vs M9 + OSF
$\psi$ contours with LCFS, O-point, X-points (saddle finder; efm X-points on the holdout), equal-$T_e$
chord pairs, residual maps, $\psi$/$q_{95}$/$\beta_N$ error histograms, $q_{95}$/$\beta_N$/per-frame
$R^2_\psi$ traces, a time scrubber and per-frame scorecards.

## 7. FEC scoring (v3.1.0)

$$
S = 0.55\,\max(0,R^2_\psi) + 0.15\,\max\!\Big(0,\tfrac{R^2_{q_{95}}+R^2_{\beta_N}}{2}\Big)
 + 0.10\,\big(1-\min(1,D_{\rm LCFS})\big) + 0.20\cdot\tfrac17\sum_{j=1}^{7}\max(0,R^2_{{\rm cons},j})
$$

* $R^2=1-SS_{\rm res}/SS_{\rm tot}$, **pooled over all frames (and pixels) of the fold**, $SS_{\rm tot}$
  about the ground-truth mean.
* One fold-global flux sign $s\in\{\pm1\}$ minimising $SS_{\rm res}(\psi)$; contour work then uses
  `axis_sign` $=\sigma_{\rm ax}s$.
* $D_{\rm LCFS}$: per frame $\min(1,\,d_H(\Gamma_{\rm pred},\Gamma_{\rm EFIT})/R_{\rm geo})$
  (symmetric Hausdorff, $R_{\rm geo}$ = mean reference major radius of the shot), averaged per shot,
  then over shots; failed predicted LCFS ⇒ 1; frames with a failed reference LCFS are dropped.
* Consistency: the seven scalars $R_{\rm axis},Z_{\rm axis},\kappa,\delta_{\rm top},\delta_{\rm bot},V,\ell_i$
  are derived by the *same* code from predicted and EFIT $\psi$; non-finite derivations use the GT mean.
* Non-finite predicted pixels/scalars are replaced by the GT mean.

The organisers' modules are vendored byte-identical in `fec_scoring/`; `pipeline/score.py` ports
the reduction (`local_score.py` / `mast_eval.score_holdout`).

## 8. Held-out results

**Holdout definition.** The M9 *sealed-80*: sorted FAIR-MAST ingest keys minus the 3 demo shots,
`random.Random(42).shuffle`, first 80 (`data/fair_mast/holdout_seed42.json`). Re-cut to the FEC
protocol: EFIT clock restricted to diverted slices (any finite efm X-point — reproduces the FEC frame
selection on all 3 demo shots), magnetics/Thomson clocks whole, GT $\psi$ = EFIT++ `psirz`, GT
$q_{95},\beta_N$ = efm `q_95`, `betan`; edge Thomson overlaid from `aye` where the ingest lacked it
(60 of 65 shots). **15 of the 80 shots have no diverted slice and are excluded → 65 shots, 3,449
frames** (of 5,280 EFIT slices). Disjoint from all M9 training: the 3,622-shot training list
∩ sealed = ∅ (proof in `make_holdout.py`; checkpoints record `seed=42`, `train_shots=3622`).
OSF caveat: its parameters were not fitted to these shots' labels, but its global $q_{95}/\beta_N$
factors were tuned on public-test leaderboard probes, and that fold may overlap the archive.

Measured on 2026-11-18 (seed 42, 65 shots, 3,449 frames, scorer v3.1.0, $s=+1$, 0 LCFS failures):

| sealed-80 holdout (65 shots, 3,449 frames) | **S** | R²ψ | R²q95 | R²βN | 1−D_LCFS | Consistency |
|---|---|---|---|---|---|---|
| OSF | **0.618634** | 0.917210 | 0.164457 | 0.379077 | 0.734036 | 0.000000 |
| M9 + OSF | **0.862677** | 0.997771 | 0.342298 | 0.394260 | 0.945207 | 0.820703 |

Consistency components (unclipped pooled $R^2$; the term averages $\max(0,\cdot)$):

| sealed-80 holdout | R²(R_axis) | R²(Z_axis) | R²(κ) | R²(δ_top) | R²(δ_bot) | R²(V) | R²(ℓ_i) |
|---|---|---|---|---|---|---|---|
| OSF | −0.492257 | −0.137536 | −0.069541 | −0.502441 | −0.169103 | −2.645417 | −0.278707 |
| M9 + OSF | 0.889709 | 0.904550 | 0.893904 | 0.906330 | 0.577001 | 0.904904 | 0.668520 |

For reference only — the 3 FEC demo shots (115 frames) were used in the OSF calibration, so they
are not held out for OSF:

| FEC demo (3 shots, 115 frames) | S | R²ψ | R²q95 | R²βN | 1−D_LCFS | Consistency |
|---|---|---|---|---|---|---|
| OSF | 0.774217 | 0.982004 | 0.841487 | 0.689767 | 0.876086 | 0.158309 |
| M9 + OSF | 0.871921 | 0.998555 | 0.818390 | 0.594122 | 0.955503 | 0.606137 |

Pooled over **all** 8,161 EFIT slices of the 80 sealed shots (ramps and limited phases, plus the
QC-quarantined outlier shot 27373), the shipped ensemble has a pixel $R^2$ of 0.378687 (0.733818
without 27373); the difference from 0.997771 is the frame population.

### 8.1 FAIR-MAST-only retraining (E1–E5)

Protocol fixed in advance in `docs/TRAINING_REGIME.md` §9; run with `bash scripts/run_experiments.sh`
(about 2.5 h on one RTX 5060 Ti); outputs in `outputs/experiments/`, paper Section 10.6. E1 = shipped
recipe (coefficient MSE) on the cleaned pool; E2 = + boundary band; E3 = + scalar heads on EFIT++
q95, βN, axis, ℓi; E4 = + critical-point and temporal terms (final pick, lowest route score on the
archive validation split); E5 = OSF q95/βN read-out maps fitted to EFIT++ on 200 training shots
(replacing c\*, s\*). Nothing in E1–E5 uses FEC labels or leaderboard probes.

| sealed-80 holdout, scored once | **S** | R²ψ | R²q95 | R²βN | 1−D_LCFS | Consistency | tip gap ≥ 50 mm | p95 Hausdorff (mm) |
|---|---|---|---|---|---|---|---|---|
| shipped M9 + OSF | **0.862677** | 0.997771 | 0.342298 | 0.394260 | 0.945207 | 0.820703 | 24.2% | 108.6 |
| E1 (MSE only) | **0.874196** | 0.997808 | 0.444019 | 0.391840 | 0.944528 | 0.841297 | 24.9% | 104.2 |
| E4 (route-guided, pick) | **0.931421** | 0.997837 | 0.916470 | 0.689018 | 0.945939 | 0.838024 | 23.1% | 97.9 |
| shipped OSF | **0.618634** | 0.917210 | 0.164457 | 0.379077 | 0.734036 | 0.000000 | 90.5% | 585.7 |
| OSF + E5 maps | **0.640558** | 0.917210 | 0.444019 | 0.391840 | 0.734036 | 0.000000 | 90.5% | 585.7 |

| FEC demo (3 shots, 115 frames) | **S** | R²ψ | R²q95 | R²βN | 1−D_LCFS | Consistency | tip gap ≥ 50 mm | p95 Hausdorff (mm) |
|---|---|---|---|---|---|---|---|---|
| shipped M9 + OSF | **0.871921** | 0.998555 | 0.818390 | 0.594122 | 0.955503 | 0.606137 | 20.0% | 64.8 |
| E1 (MSE only) | **0.863300** | 0.998826 | 0.635596 | 0.576591 | 0.949921 | 0.640195 | 31.3% | 87.5 |
| E4 (route-guided, pick) | **0.867116** | 0.998764 | 0.975269 | 0.417430 | 0.952618 | 0.590407 | 22.6% | 80.6 |

E4 beats the shipped hybrid on the holdout and is slightly below it on the three demo shots, where
the shipped scalar read-out was tuned. The gain over E1 comes from the trained scalar heads; the
geometric route terms did not improve the boundary beyond MSE (route score E4 0.9906 vs E1 0.9742
relative to shipped on the holdout). Tables: `python scripts/make_experiment_tables.py`.

## 9. Verification

| check | result |
|---|---|
| OSF port vs original `osf/mast_12_osf_0_0_12.py` (3 demo shots) | max \|Δψ\|, \|Δq95\|, \|ΔβN\| = 0 |
| M9 port vs original `generate_final_predictions` path (3 demo shots) | max \|Δψ\| = 0 |
| `pipeline.score` vs catskills_fec `mast_eval.score_holdout` on the same predictions | all 19 result fields identical (holdout osf/hybrid, demo osf/hybrid) |
| `models.m9.train` split / QC / window | QC keep == shipped 3,622 list; ip_lo == 328.7293519015368 |
| viewer | all routes 200, unknown routes 404 |

Re-run: `python tests/parity_demo.py <ref_dir>`, `python tests/crosscheck_dump_gt.py …`,
`tests/crosscheck_catskills_scorer.py` (needs the original repository).

## Build the paper

The paper is a tree of short Markdown notes under `paper/` (root `paper/MAST.md`; each bullet link is
a child note, one link hop = one section level), readable on their own and compiled to LaTeX/PDF:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/make_result_figures.py  # figures + tables from outputs/
.venv/bin/python scripts/make_audit_tables.py      # audit tables from outputs/audit/*_shipped
python3 scripts/compile_md_tree.py      # notes -> paper/arxiv.md -> paper/arxiv.tex -> paper/MAST.pdf
bash scripts/build_paper_pdf.sh         # re-flatten and rebuild the PDF only
```

`compile_md_tree.py --md-only` / `--tex-only` stop early. Needs `pip install -e .[paper]`, pandoc,
latexmk/pdflatex and Graphviz (`dot -Tpdf paper/figures/src/X.dot` re-renders a diagram). The build
fails on any undefined citation or reference. Result tables in `paper/generated/` are written by
`make_result_figures.py` and `make_audit_tables.py` from the saved `outputs/*.score.json` and
`outputs/audit/*/summary.json`, so the paper never hand-copies a
number.

## Repository layout

```
data/fec/            fetch.py (demo + public test), README
data/fair_mast/      ingest.py, make_holdout.py, fetch_efit_labels.py, holdout_seed42.json, QC manifests
fec_scoring/         organisers' scorer modules (byte-identical) + mast_envelope.npz
models/io.py         parquet row I/O, EFIT-frame cut
models/osf/          primitives.py, solver.py, calibration.py
models/m9/           features.py, network.py, predict.py, train.py
pipeline/            paths.py, datasets.py, predict.py, score.py
app/fec_site/        MAST viewer (stdlib HTTP server + Plotly)
docs/MODEL.md        derivations and parameter provenance
tests/               parity and cross-check scripts
```

Licence: Apache License 2.0 (see `LICENSE` and `NOTICE`); Copyright 2026 Lars Warren Ericson (d/b/a Catskills Research Company). Data licences: FEC CC BY 4.0; FAIR-MAST CC BY-SA 4.0.
