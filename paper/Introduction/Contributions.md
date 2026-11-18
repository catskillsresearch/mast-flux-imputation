# Contributions

1. A two-part objective for flux imputation: the challenge score as target, and frame-by-frame
   observable distances (field, boundary, O/X-points, scalar traces) as route, with the hypothesis
   that the route is needed for the target to generalise (Section 7).
2. An automated observable audit, `pipeline/audit.py`, that computes the route quantities for every
   frame of any FAIR-MAST shot, reproduces the official composite score, and reports the measured
   gaps between target and route for the existing models (Section 7.5).
3. A complete mathematical description of OSF (`mast_12_osf_0.0.12`), a free-boundary
   Grad–Shafranov forward solver driven by measured currents, separating machine constants from the
   38 calibrated parameters (Section 3).
4. An exact description of M9 (`M9_win50`), an ensemble of three PCA-basis neural surrogates of
   EFIT++ flux trained on FAIR-MAST [@jackson2024fairmast] with coefficient MSE; it is not a
   physics-informed neural network (Section 4).
5. A reproducible build: data acquisition, ingest, quality control, holdout construction with a
   disjointness proof, training, prediction, scoring and a browser viewer (Section 5), with measured
   results on a sealed holdout re-derived from saved prediction files (Section 8).
6. A record of the data each component was trained or tuned on (Section 9.2), and a planned
   FAIR-MAST-only regime that tests the thesis directly (Section 10).
