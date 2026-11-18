# M9

Archive-trained $\psi$ predictor (3-member ensemble, PCA-60, dual-context residual MLP).

* Weights (not in git): `CRC_FEQ_mast_M9_win50_i{101,202,303}.pt` in `models/m9/weights/` or
  `$M9_WEIGHTS_DIR`. `M9Ensemble(verify_sha=True)` checks the sha256 of the shipped files.
* Inference: `M9Ensemble(dir).predict(row)` → (T, 65, 65), mean of the members.
* Training: `python -m models.m9.train --build-cache` then `--member all` (smoke test:
  `--train-shots 40 --epochs 1 --out /tmp/m9smoke`). A full retrain was not re-run during the
  extraction; the split, QC keep list and $|I_p|$ window were reproduced exactly.
* Training data are FAIR-MAST EFIT++ reconstructions (see DATA_PROVENANCE.md).
