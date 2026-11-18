# FAIR-MAST-only training regime

Design for retraining and recalibrating the MAST flux models from FAIR-MAST data alone, guided
by the observable audit (`pipeline/audit.py`), and for testing the paper's thesis: a simple loss
(coefficient MSE) gives a high target score on fitting/selection data but generalises less well
than training guided by observable distances. Nothing here has been trained yet; all times are
estimates calibrated by one smoke run (§8).

## 1. Acceptance test (old vs new)

"Old" is the shipped calibration (M9_win50 ensemble + OSF `mast_12_osf_0.0.12`, c\*, s\*), audited
with the same code:

```bash
python -m pipeline.audit outputs/<split>_<mode>.npz --tag <split>_<mode>_shipped   # done
python -m pipeline.audit outputs/<split>_<mode>_new.npz --tag <split>_<mode>_new
python -m pipeline.audit compare outputs/audit/<split>_<mode>_shipped outputs/audit/<split>_<mode>_new
```

`compare` checks every entry of `SUMMARY_SPEC` in its own direction (S, R²ψ, R²q95, R²βN,
1−D_LCFS, Consistency and its 7 terms, frame R²ψ, MSE, mean/p95 Hausdorff, area mismatch, O- and
X-point offsets, LCFS-fail / X-point-lost / topology-flip / tip-gap rates, q95/βN/κ/ℓi MAE, lag,
jitter). A new calibration is accepted if it is at least as good on every metric on the three
validation shots, and then once on the sealed holdout. Baseline (shipped) values:

| metric | M9+OSF validation | M9+OSF holdout | OSF validation | OSF holdout |
|---|---|---|---|---|
| S | 0.871921 | 0.862677 | 0.774217 | 0.618634 |
| R²ψ | 0.998555 | 0.997771 | 0.982004 | 0.917210 |
| R²q95 / R²βN | 0.818390 / 0.594122 | 0.342298 / 0.394260 | 0.841487 / 0.689767 | 0.164457 / 0.379077 |
| 1−D_LCFS | 0.955503 | 0.945207 | 0.876086 | 0.734036 |
| Consistency | 0.606137 | 0.820703 | 0.158309 | 0.000000 |
| mean / p95 Hausdorff (mm) | 35.6 / 64.8 | 41.3 / 108.6 | 98.6 / 215.9 | 207.2 / 585.7 |
| tip-gap ≥ 50 mm rate | 20.0% | 24.2% | 73.0% | 90.5% |
| limited/diverted flip rate | 0.0% | 2.9% | 0.0% | 6.2% |

Two practical caveats. With three shots, some validation R² values are dominated by tiny
between-shot variance (R²(Z_axis) = −60.08), so "≥ old" on those terms is noisy; we also report
the archive validation split (§4) and treat a validation-shot regression within one bootstrap
standard error (resampling frames within shots) as a tie, stated explicitly. And OSF's q95/βN
in the hybrid used c\*, s\* fitted from leaderboard probes; a FAIR-MAST-only refit (E5) is the
hardest item to match.

## 2. Data findings that shape the regime

* **Labels.** For 28348/28350/28351, FAIR-MAST EFIT++ ψ, q95, βN, βp, ℓi, κ, axis, ψ_axis/ψ_b
  and LCFS are bit-identical to the FEC release (`outputs/compat/fair_mast_labels.json`). In a
  50-shot random sample of the training pool, 43 shots carry the full `efm` product (ψ on 65×129
  with 65 finite R columns, all scalars 100% finite on diverted frames); 7 do not (next item).
* **No-plasma shots.** 349 of 3,738 ingested shots have a reduced EFIT++ product: exactly 200
  slices on a fixed 5 ms clock (t0 = 0 or −50 ms), 60 `efm` variables instead of ~150, no
  scalars or X-points, ψ range median 0.08 Wb/rad (normal shots 0.73), and max|Ip| ≤ 46.8 kA
  (median 9.1 kA). They are shots without plasma, spread over 23240–30442. 335 are in the QC
  keep list, 14 are in the sealed 80 (14 of the 15 sealed shots with no diverted slice).
* **Did they enter M9?** Not as training frames and not in the PCA basis: none has a frame above
  the |Ip| window (0 of 335; the 3,265 contributing shots are all normal; 22 normal shots also
  contribute nothing). They did enter the QC median/MAD statistics and the 2-means fit of the
  window: ip_lo = 328.729 kA with them, 471.990 kA without. The shipped window therefore depends
  on them.
* **Input format** (archive ingest vs FEC rows): magnetics clock offset 0.1 ms (Ip differences
  up to 44–63 kA in fast transients, not at EFIT frames); no edge Thomson in the ingest (zeroed
  at inference only); 131 vs 130 core radii.

Scan scripts: `scripts/check_fair_mast_labels.py` (S3 metadata, 50-shot sample) and
`scripts/scan_no_plasma_shots.py` (ingest `efit_times` and parquet column statistics, cache Ip
column only; writes `outputs/compat/no_plasma.json`).

## 3. Data selection

1. **Exclude no-plasma shots explicitly** by signature (efit T = 200 on the fixed 5 ms clock and
   no `efm/q_95`) and by max|Ip| < 50 kA; record the list in `data/fair_mast/no_plasma.txt`.
2. **Fix the |Ip| window** at the shipped 328.7293519 kA as a named constant in both arms (no
   refit), so the arms differ only in loss and selection and the frame population stays that of
   the shipped model. A separate ablation can test 472 kA.
3. **QC** as shipped (max|ψ| > median + 10 MAD, short / non-finite / constant), recomputed
   without the no-plasma shots; record the diff to the shipped keep list.
4. **Stratification** for sampling and reporting: shot-number band (23–24k, 24–26k, 26–28k,
   28–30k, 30k+), diverted vs limited frame, Ip band. Shot-balanced sampling as shipped.
5. **Splits:** sealed 80 untouched (seed 42); 10% of training shots (seed 42) as archive
   validation; the three FEC validation shots as a second validation set, never trained on.
6. **Labels for route terms:** LCFS, O/X-points, κ, δ, V, ℓi are computed offline from the
   cached EFIT ψ with the audit's own extractors (no S3); q95 and βN come from
   `data/fair_mast/fetch_efit_labels.py` for all training shots.

## 4. Input fixes (both arms)

1. Resample the magnetics onto the FEC clock (remove the 0.1 ms offset) in the ingest.
2. Edge Thomson: zero the edge columns in training *and* inference (the ingest has none), or
   re-ingest `aye` for all shots and use it in both; the first is the default.
3. Core Thomson radii: use the 130-entry FEC table (affects OSF βN only).
4. Rebuild the feature cache once with these fixes (`--build-cache`, repo-local, not on the lab
   drive).

## 5. Paired experiment: arm A vs arm B

Same data (§3), inputs (§4), network (`dualctx` 512×4, context window 9), PCA basis (K = 60),
seeds (101/202/303), epochs (50), optimiser and batch schedule in both arms.

* **Arm A (MSE):** loss = MSE on the 60 PCA coefficients; checkpoint = best archive-validation
  coefficient MSE; ensemble accepted if validation-shot R²ψ ≥ 0.9976 (the shipped rule).
* **Arm B (route-guided):** arm A's loss plus, on the reconstructed ψ̂ = mean + Σ c_k φ_k:
  1. *Boundary band:* pixel MSE weighted ×w_b inside |ψ_N − 1| < 0.1 of the EFIT LCFS.
  2. *Soft LCFS area:* 1 − soft-IoU between σ((1 − ψ̂_N)/τ) and the EFIT interior mask (ψ̂_N
     normalised with EFIT ψ_axis, ψ_b), τ ≈ 0.02.
  3. *Critical points:* |∇ψ̂|² at the EFIT O-point and X-point(s) (bilinear interpolation of
     finite differences), and (ψ̂(x_X) − ψ̂_b)².
  4. *Scalar heads* (already in the network, untrained today) on q95, βN, R_axis, Z_axis, ℓi
     with Huber loss on standardised targets.
  5. *Temporal:* MSE between predicted and EFIT frame-to-frame coefficient differences within
     each context window.
  Checkpoint selection by the audit on the archive validation split: lexicographic gate
  (tip-gap rate, p95 Hausdorff, flip rate, then S) instead of coefficient MSE.
* **Report** for both arms: audit tables on archive validation, the three validation shots, and
  (once, at the end) the sealed holdout; the generalisation gap validation→holdout per metric.
  The thesis predicts B ≥ A on holdout S and on the route metrics, with a smaller gap.

## 6. Ranked experiments (by expected value)

GPU: RTX 5060 Ti 16 GB. Times are estimates from the smoke run (§8).

| rank | experiment | why | GPU time |
|---|---|---|---|
| 1 | **E1 arm A control**: shipped recipe on the cleaned data (§3–4), 3 members, audit | Control for the thesis test; checks that the cleanup reproduces the shipped scores | ~35 min (3 × ≤12.5 min) + audit ~2 min |
| 2 | **E2 arm B, terms 1–2** (boundary band + soft LCFS), small weight sweep on 1 member (4 settings × 10 epochs), then 3 members | Targets the dominant failure (20–24% tip-gap frames) | sweep ~25 min; 3 members ~1–1.5 h (route terms ~1.5–2× step cost) |
| 3 | **E3 scalar heads + audit checkpoint gate** on top of the best E2 | q95/βN are the weakest target terms (holdout R² 0.34 / 0.39) and today depend on c\*, s\* | ~1–1.5 h |
| 4 | **E4 critical-point + temporal terms** on top of E3 | X-point offset 7.9→21.2 mm and 2.9% flips on holdout; lag/jitter | ~1.5 h |
| 5 | **E5 OSF FAIR-MAST-only recalibration**: fit the calibrated group and c\*, s\* by an explicit optimiser on archive training shots with EFIT++ scalars | Removes leaderboard-probe dependence; must match old q95/βN | mostly CPU; to be timed on 200 shots before committing to a full run |

Total GPU for E1–E4 ≈ 4–5 h. Run E1 first; do not open the sealed holdout until the final E1-vs-best-B comparison.

## 7. What the earlier MAST runs on the lab drive show

From `results/mast_archive/P36_*.json` (Sept 15 and Oct 30, 2026). "demo" there means the three
validation shots; "sealed" is pooled over *all* EFIT slices of the sealed 80 (including
no-plasma and limited frames), not the diverted-only protocol used here.

| run | change | demo R²ψ | sealed all-slice R²ψ (clean 79) |
|---|---|---|---|
| M1_archive / M1_long | archive baseline, 3,655 shots | 0.980803 / 0.981548 | 0.513 |
| M2_vacres | predict residual from vacuum field | 0.975249 | 0.512 (worse) |
| M3_qc | QC fence + shot-balanced sampling | 0.980459 | 0.509 (0.904) |
| M4_edge | edge handling | 0.993369 | 0.506 (0.896) |
| M6_m3long | edge mask, 50 epochs | 0.995908 | 0.507 (0.898) |
| M7_win | Ip window, 12 epochs | 0.997596 | 0.377 (0.731) |
| M8_winlong | window, 50 epochs | 0.997730 | 0.380 (0.736) |
| M9_win100_i42 | 100 epochs | 0.998098 | 0.380 (0.736) |
| M9_win50_i101/202/303 | shipped members | 0.998571 / 0.998406 / 0.998172 | 0.378 / 0.379 / 0.377 |

What worked for the selection metric: QC + shot balancing, the edge mask, 50 epochs, the |Ip|
window, three-seed ensembling. What did not: the vacuum-residual target (M2), 100 vs 50 epochs
(no gain). The |Ip| window raised validation R²ψ from 0.9959 to 0.9976 while all-slice sealed
R²ψ fell from 0.898 to 0.731 (clean): the gate on three shots' diverted frames selected a model
that is worse outside that population. Selection on a narrow target metric is the pattern the
thesis describes; arm B's gate uses the audit on a 10% archive split instead.

Other MAST material on the drive: `olive_geom/mast_pca60_ceiling.json` (leave-one-shot-out
PCA on the three validation shots, worst R²ψ 0.99846: the 60-mode basis is not the bottleneck);
`olive_geom/mast_freegs_*` (FreeGS as a teacher failed, worst R²ψ 0.42; a FreeGS Picard
calibration reached mean R² 0.926); `mast_qb/qb_rules_*` (rule-based q95/βN gave negative R²);
`benchmarks/CRC_FEQ_osf/` (OSF versions 0.9.1–2.1.0 and the c\*/s\* probe submissions);
`benchmarks/CRC_FEQ_phase3/` (reproduction tarballs and provenance). Data: `datasets/fair_mast_train`
(3,738 ingest parquets, 27 GB), `cache/mast_features_v1` (6.1 GB, X and EFIT ψ per shot),
`cache/mast_qc_v1`, `cache/mast_win_v1` (QC and window indexes), FEC release (102 GB, both
machines), shipped M9 weights in `artifacts/`. Most other result files are DIII-D and not used.

## 8. Cost calibration and memory discipline

Smoke run (Nov 18, 2026, 17:50 ET): `python -m models.m9.train --feature-cache <lab>/cache/mast_features_v1
--member 101 --epochs 2 --out /tmp/...` reproduced the shipped split, keep list (True), ip_lo
(328.7293519 kA) and windowed population (211,935 frames, 3,265 shots); 2 epochs took 30 s of
training (55 s wall including load, QC, window and PCA) with peak RSS 14.2 GB. Hence ≤ 12.5 min per
50-epoch member. Run members sequentially, `OMP_NUM_THREADS=4`, audit with ≤ 8 workers and
chunked shots; keep all outputs off the lab drive.
