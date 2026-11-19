# FAIR-MAST-only training regime

How the MAST hybrid M9 + OSF is trained and calibrated from FAIR-MAST data alone, guided by the
observable audit (`pipeline/audit.py`), and how the paper's thesis is tested: a simple loss
(coefficient MSE) gives a high target score on fitting/selection data but generalises less well
than training guided by observable distances. §1 lists the data findings that shape the regime,
§2 the data selection, §3 the protocol of E1–E5 (fixed before the run), §4 their results, §5 the
E6 curve-distance run.

## 1. Data findings that shape the regime

* **Labels.** For 28348/28350/28351, FAIR-MAST EFIT++ ψ, q95, βN, βp, ℓi, κ, axis, ψ_axis/ψ_b
  and LCFS are bit-identical to the FEC release (`outputs/compat/fair_mast_labels.json`). In a
  50-shot random sample of the training pool, 43 shots carry the full `efm` product (ψ on 65×129
  with 65 finite R columns, all scalars 100% finite on diverted frames); 7 do not (next item).
* **No-plasma shots.** 349 of 3,738 ingested shots have a reduced EFIT++ product: exactly 200
  slices on a fixed 5 ms clock (t0 = 0 or −50 ms), 60 `efm` variables instead of ~150, no
  scalars or X-points, ψ range median 0.08 Wb/rad (normal shots 0.73), and max|Ip| ≤ 46.8 kA
  (median 9.1 kA). They are shots without plasma, spread over 23240–30442. 335 are in the QC
  keep list, 14 are in the sealed 80 (14 of the 15 sealed shots with no diverted slice).
* **Do they enter M9?** Not as training frames and not in the PCA basis: none has a frame above
  the |Ip| window (0 of 335; the 3,265 contributing shots are all normal; 22 normal shots also
  contribute nothing). They did enter the QC median/MAD statistics and the 2-means fit of the
  window: ip_lo = 328.729 kA with them, 471.990 kA without, so the window is fixed as a named
  constant (§2).
* **Input format** (archive ingest vs FEC rows): magnetics clock offset 0.1 ms (Ip differences
  up to 44–63 kA in fast transients, not at EFIT frames); no edge Thomson in the ingest (zeroed
  at inference only); 131 vs 130 core radii.

Scan scripts: `scripts/check_fair_mast_labels.py` (S3 metadata, 50-shot sample) and
`scripts/scan_no_plasma_shots.py` (ingest `efit_times` and parquet column statistics, cache Ip
column only; writes `outputs/compat/no_plasma.json`).

## 2. Data selection

1. **Exclude no-plasma shots** by signature (EFIT T = 200 on the fixed 5 ms clock, no `efm/q_95`)
   and by max|Ip| < 50 kA (`outputs/compat/no_plasma.json`).
2. **Fix the |Ip| window** at 328.7293519 kA as a named constant in every experiment (no refit),
   so the experiments differ only in loss and selection. A separate ablation can test 472 kA.
3. **QC** (max|ψ| > median + 10 MAD, short / non-finite / constant), recomputed without the
   no-plasma shots: 3,280 kept; 3,258 shots contribute 211,534 windowed frames.
4. **Splits:** sealed 80 untouched (seed 42); 10% of the contributing shots (seed 42, 325 shots) as
   archive validation; the three FEC validation shots as a second validation set, never trained on.
5. **Labels for route terms:** LCFS, O/X-points, κ, δ, V, ℓi are computed offline from the cached
   EFIT ψ with the audit's own extractors; q95 and βN come from `data/fair_mast/fetch_efit_labels.py`.

Scan scripts: `scripts/check_fair_mast_labels.py` (S3 metadata, 50-shot sample) and
`scripts/scan_no_plasma_shots.py` (writes `outputs/compat/no_plasma.json`).

## 3. Protocol for the E1–E5 run (fixed before any result)

Code: `experiments/` (`prep`, `route_labels`, `m9x`, `osf_recal`, `evaluate`, `driver`), launched by
`scripts/run_experiments.sh`; progress in `outputs/experiments/STATUS.json`, logs in
`outputs/experiments/<stage>/`.

* **Inputs.** Every experiment reads the same feature cache, so all share inputs exactly; the
  edge-Thomson columns are zeroed.
* **Data.** Sealed-80 split; pool minus the no-plasma shots; QC recomputed; |Ip| window fixed at
  328.7293519 kA; one PCA basis (K = 60) shared by every experiment and member. Archive validation
  = first 10% of the contributing shots after `random.Random(42).shuffle`; `archval100` and
  `select30` are its first 100 and 30 shots; `osftrain200` = 200 training shots
  (`random.Random(43)`). Archive shots are evaluated with the FEC protocol cut (diverted EFIT
  frames, `pipeline.datasets.load_holdout`).
* **Route score.** R = mean over {tip-gap rate, p95 and mean Hausdorff, relative area symmetric
  difference, X-point and O-point offsets, topology-flip rate (+0.005)} of the ratio new/reference,
  with the reference model audited on the same shots. R < 1 is better than the reference. Sweeps
  and checkpoint choices on `select30` use a fixed M9 ensemble trained with coefficient MSE as the
  reference; the result tables report R relative to E1.
* **E1 (MSE only).** Coefficient MSE; per member the epoch with the lowest archive-validation
  coefficient MSE; three members (101/202/303), 50 epochs. Scalars: E5 OSF maps.
* **E2 (+ band, soft LCFS).** Sweep on member 101, 10 epochs, (w_b, w_a) ∈ {(1, 0), (3, 0),
  (1, 3·10⁻³), (3, 10⁻²)}; pick the lowest R on `select30`. Then three members × 50 epochs;
  candidate states at epochs 19, 29, 39, 44, 49 and the best-MSE epoch; per member the candidate
  with the lowest R on `select30`. Scalars: E5 OSF maps.
* **E3 (+ scalar heads).** E2's weights plus w_s ∈ {3·10⁻⁴, 3·10⁻³} (Huber on standardised q95,
  βN, R_axis, Z_axis, ℓi from EFIT++); pick the highest R²qb on `select30` among settings with R
  within 5% of the lowest. Members and checkpoint rule as E2. Scalars: the network heads.
* **E4 (+ critical points, temporal).** E3's weights plus (w_c, w_t) ∈ {(10⁻³, 1), (10⁻², 1)};
  lowest R on `select30`. Members and checkpoint rule as E2. Scalars: the network heads.
* **E5 (OSF read-out maps).** OSF solved on `osftrain200` (FEC protocol cut); affine maps
  q95 ← clip(a_q q + b_q, 2, 25), βN ← clip(a_b β + b_b, 0.2, 5) fitted by Nelder–Mead (start: the
  identity) to the EFIT++ labels of those frames. The 38 solver parameters are not refitted.
* **Acceptance tables.** For every experiment `pipeline.audit compare` against E1 on the three
  validation shots and on `archval100`.
* **Final pick.** Among the E2–E4 ensembles, the lowest R on `archval100`; tie (equal to 4
  decimals) broken by validation-shot S.
* **Sealed holdout, once.** After the pick: E1, the final pick and E5 OSF mode. The thesis predicts
  the pick ≥ E1 on holdout route metrics and S.

## 4. Results of the E1–E5 run (Nov 18, 2026)

Run with `bash scripts/run_experiments.sh` on one RTX 5060 Ti 16 GB. All tables are generated by
`scripts/make_experiment_tables.py` into `paper/generated/exp_*.md`.

* **Sweeps** (`exp_sweeps.md`): E2 picked (w_b, w_a) = (3, 0) (the soft-LCFS term raised R at both
  weights); E3 picked w_s = 3·10⁻³; E4 picked (w_c, w_t) = (10⁻³, 1).
* **Validation shots** (`exp_demo.md`): S = 0.863300 (E1), 0.858226 (E2), 0.864866 (E3),
  0.867116 (E4).
* **archval100** (`exp_archval100.md`): R relative to E1 = 1.0135 (E2), 0.9874 (E3), 0.9557 (E4);
  final pick **E4**.
* **E5** (`exp_e5.md`): maps a_q = 0.626, b_q = 1.957, a_b = 1.013, b_b = 0.093; OSF S on the
  holdout 0.640558, on the validation shots 0.750287.
* **Sealed holdout, once** (`exp_holdout.md`): S = 0.874196 (E1), **0.931421 (E4)**; R of E4
  relative to E1 = 1.0186 (better tip-gap rate and Hausdorff distances, worse O/X-point offsets and
  flip rate).
* **Thesis.** The pick beats E1 on holdout S (from the scalar heads) but not on the composite route
  score; the geometric route terms did not improve the boundary beyond coefficient MSE.

## 5. E6: curve-distance calibration

Code: `experiments/e6.py` (trainer, differentiable read-out, objective), `experiments/driver_e6.py`
(stages truth → s1 → sweep → m101…m505 → select → final, resumable; status in
`outputs/experiments/E6/STATUS.json`), launched by `scripts/run_e6.sh` (systemd unit, `status`
subcommand). Tables and figures: `scripts/make_e6_report.py` → `paper/generated/e6_*.md`,
`paper/figures/e6_*`.

* **Loss.** PCA-coefficient MSE anchor; w_cd × range-normalised per-shot curve distance (RMSE plus
  0.5 × the same on frame differences) on q95 and βN from the heads and on κ, δ_top, δ_bot, V,
  R_axis, Z_axis from a differentiable read-out of ψ̂; w_lcfs = w_cd/3 × (1 − soft IoU); boundary
  band w_b = 3; tip band w_tip = 10 (top and bottom 15% of the LCFS height); O/X-point term
  w_c = 10⁻³; temporal term w_t = 1; head Huber w_s = 3·10⁻³. Late weight 1 + α·ramp over the last
  20% of each shot (α = 4 for βN, 2 otherwise); series weight 2 for βN, δ_top, δ_bot.
* **Batches and optimiser.** 32 whole shots per step, at most 128 frames per shot; AdamW 10⁻³,
  ReduceLROnPlateau (0.5, patience 4, floor 2·10⁻⁵), EMA 0.998.
* **Objective J** on `archval100`: the same weighted curve distances plus the late-weighted
  1 − soft IoU of the LCFS.
* **Spike guard.** If J > 1.2 × best: return to the best EMA weights and halve the learning rate.
* **Stopping.** No gain over 0.2% on the best J for 20 epochs AND J less than 1% below its value 5
  epochs earlier; at least 30, at most 300 epochs. Keep the top-3 EMA states.
* **Step 1 (re-selection).** Re-selecting the existing E3/E4 checkpoints by J: archival J 0.15321 →
  0.14643, S 0.934308 → 0.933418 (`e6_selection.md`), so new training is needed.
* **Sweep** (seed 101, 40 epochs): w_cd ∈ {0.01, 0.03, 0.1, 0.3}; picked 0.01 (best J 0.15583).
* **Seeds** 101/202/303/404/505 stopped at epochs 103, 172, 127, 142, 106 (best-J epochs 82, 152,
  107, 122, 86), each after one or two spike-guard returns.
* **Selection.** Rank-k ensembles (k-th best EMA state of each seed), first with archval100 S ≥ E4's
  0.934308: rank 1 S 0.933913 (fails), rank 2 S 0.934929 (picked).
* **Scored once.** Validation shots S = 0.899567 (E4 0.867116); sealed holdout S = 0.934444 (E4
  0.931421), Consistency 0.862171, R²(δ_bot) 0.761293, R²(Z_axis) 0.961958; tip-gap rate 32.4%,
  X-point offset 34.3 mm, 1 − D_LCFS 0.938632 (E4: 23.1%, 22.0 mm, 0.945939).
* **Reading.** The curve-distance terms improve the shape scalars and the score on unseen shots and
  cost boundary position: the LCFS and X-point carry little weight next to the shape scalars, the
  soft κ read-out's errors agree only moderately with the scorer's (r = 0.434), and ℓi has no
  curve-distance term.
