# Contributions

1. A flux-imputation model for MAST: a 60-mode principal-component basis of EFIT++ flux, an
   ensemble of five residual networks with a gated temporal-context branch, and linear scalar heads
   for $q_{95}$ and $\beta_N$ (Model section).
2. A calibration that trains the network on per-shot curve distances of the scalar time series,
   reading the shape scalars from the predicted flux through differentiable extractors, and selects
   checkpoints on archive validation shots (Calibration section).
3. A build from public data only: FAIR-MAST download and ingest, quality control, a sealed holdout
   with a disjointness check, training, prediction, scoring and a browser viewer.
4. Ground-truth-against-prediction results on a sealed archive holdout and on three held-out MAST
   validation shots, with per-shot scores, curve distances, time series and boundary overlays
   (Results section), all regenerated from saved prediction and audit files.
