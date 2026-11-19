# Data use and tuning facts

What each component was trained, tuned or evaluated on, as established from the code and data:

1. M9's training targets and PCA basis are FAIR-MAST EFIT++ $\psi$ from the QC-kept archive pool
   (3,280 shots; 3,258 contribute frames after the $|I_p|$ window, 211,534 frames), with 10% of the
   contributing shots (325) as the validation split. M9 trains on all windowed frames, limited
   ones included.
2. The 335 no-plasma shots of the pool are removed before QC. They contribute no frame in any case
   but would set the window threshold (328.73 kA with them, 471.99 kA without), so the threshold is
   fixed at 328.7293519 kA.
3. The base loss is MSE on the 60 PCA coefficients; the route-guided runs add the terms of
   Section 10.6, and from E3 on the five scalar heads are trained on EFIT++ $q_{95}$, $\beta_N$,
   $R_{\rm axis}$, $Z_{\rm axis}$, $\ell_i$.
4. OSF's read-outs extract the LCFS with the scorer's vessel-envelope mask (`mast_envelope.npz`,
   `mask_coarse`).
5. OSF's shape parameters were chosen with reference to $R^2$ values on the three labelled demo
   shots and a small number of aggregate scores on the public-test fold; the demo shots are not
   held out for OSF.
6. The hybrid's OSF read-out maps (E5) are fitted on 200 FAIR-MAST training shots only.
7. OSF performs up to 20 Picard passes per frame (Section 3.5). Its $q_{95}$ uses the vacuum $F_b$
   and an empirical factor law (Section 3.7).
8. FAIR-MAST and FEC carry bit-identical EFIT++ $\psi$ for the same shot and time; the two differ in
   frame population (all slices versus diverted slices only).
9. No magnetic sensor is a model input; the EFIT++ labels were reconstructed from magnetic sensors.
10. The sealed-80 holdout is label-disjoint from all training, validation and read-out fitting.
11. The holdout inputs are processed by the archive ingest, which differs from the FEC rows in
    current resampling, edge-Thomson source and core radius table (Section 9.3).
12. The FAIR-MAST pool very likely shares shots with the FEC withheld test folds.

The same list, with file-level detail, is in `DATA_PROVENANCE.md` of the repository.
