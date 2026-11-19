# Contributions

1. A two-part objective for flux imputation: the challenge score as target, and frame-by-frame
   observable distances (field, boundary, O/X-points, scalar traces) as route, with the hypothesis
   that the route is needed for the target to generalise (Section 7).
2. An automated observable audit, `pipeline/audit.py`, that computes the route quantities for every
   frame of any FAIR-MAST shot, reproduces the official composite score, and reports the measured
   gaps between target and route (Section 7.5).
3. A complete mathematical description of OSF (`mast_12_osf_0.0.12`), a free-boundary
   Grad–Shafranov forward solver driven by measured currents, separating machine constants from the
   38 calibrated parameters (Section 3).
4. An exact description of M9, an ensemble of PCA-basis neural surrogates of EFIT++ flux trained
   on FAIR-MAST [@jackson2024fairmast] (Section 4), and of the hybrid M9 + OSF (Section 5).
5. A reproducible build: data acquisition, ingest, quality control, holdout construction with a
   disjointness proof, training, prediction, scoring and a browser viewer (Section 5), with measured
   results on a sealed holdout re-derived from saved audit files (Section 8).
6. A record of the data each component was trained or tuned on (Section 9.2), and a
   FAIR-MAST-only training regime that tests the thesis directly: MSE-only and route-guided M9
   ensembles and a FAIR-MAST-only OSF read-out, with the route-guided pick reaching $S=0.931421$ on
   the sealed holdout against $0.874196$ for MSE only, and a curve-distance calibration (E6)
   reaching $S=0.934444$ (Sections 8 and 10.6).
