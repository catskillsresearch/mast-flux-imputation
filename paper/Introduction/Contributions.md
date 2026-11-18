# Contributions

1. A complete mathematical description of OSF (`mast_12_osf_0.0.12`), a free-boundary
   Grad–Shafranov forward solver driven by measured currents, separating machine constants from the
   38 calibrated parameters (Section 3).
2. An exact description of M9 (`M9_win50`), an ensemble of three PCA-basis neural surrogates of
   EFIT++ flux trained on FAIR-MAST [@jackson2024fairmast], and an explicit statement that it is
   *not* a physics-informed neural network: its loss has no Grad–Shafranov residual (Section 4).
3. A reproducible build: data acquisition, ingest, quality control, holdout construction with a
   disjointness proof, training, prediction, scoring and a browser viewer (Section 5).
4. Measured results on a sealed 80-shot holdout, re-derived from the saved prediction files and
   cross-checked against an independent implementation of the scorer (Section 7).
5. An honest provenance audit, listing where earlier documentation and the code disagree, and a
   proposal for an automated failure-mode audit to sharpen training (Sections 8–9).
