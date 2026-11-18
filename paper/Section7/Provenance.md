# Data use and tuning facts

What each component was trained, tuned or evaluated on, as established from the code and data:

1. M9's training targets and PCA basis are FAIR-MAST EFIT++ $\psi$ from 3,622 archive shots. The
   training pool very likely overlaps the FEC hidden test folds, so M9 is not a zero-shot result.
2. The checkpoint metadata records `train_shots = 3622`; after the $|I_p|$ window 3,265 shots
   contribute frames, and 10% of those shots are the validation split. M9 trains on all windowed
   frames, limited ones included.
3. M9's loss is MSE on the 60 PCA coefficients only; its five scalar heads are untrained.
4. OSF's read-outs extract the LCFS with the scorer's vessel-envelope mask (`mast_envelope.npz`,
   `mask_coarse`).
5. OSF's shape parameters were chosen with reference to $R^2$ values on the three labelled demo shots,
   and the M9 run's acceptance gate is demo $R^2_\psi\ge0.9976$. The demo shots are therefore not
   held out for either model.
6. The hybrid factors $c^{*}=1.062242$ and $s^{*}=1.078508$ were fitted from leaderboard probes on
   the public-test fold.
7. OSF performs up to 20 Picard passes per frame (Section 3.5). Its $q_{95}$ uses the vacuum $F_b$ and
   an empirical factor law (Section 3.7); the hybrid map multiplies it by $c^{*}/1.20$.
8. FAIR-MAST and FEC carry bit-identical EFIT++ $\psi$ for the same shot and time; the two differ in
   frame population (all slices versus diverted slices only).
9. No magnetic sensor is a model input; the EFIT++ labels were reconstructed from magnetic sensors.
10. The sealed-80 holdout is label-disjoint from M9 training, but OSF's global factors and shape
    choices were tuned with public-test probes and the demo shots.
11. The holdout inputs are processed by the archive ingest, which differs from the FEC rows in
    current resampling, edge-Thomson source and core radius table (Section 9.3).
12. The shipped M9 weights match their recorded `SHA256SUMS`.

The same list, with file-level detail, is in `DATA_PROVENANCE.md` of the repository.
