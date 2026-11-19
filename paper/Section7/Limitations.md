# Limitations

* **Labels are magnetics-based.** EFIT++ used probes and flux loops; the task is imputation of a
  magnetics-constrained reconstruction, and its errors are inherited.
* **Archive generalisation, not zero-shot.** The FAIR-MAST training pool very likely shares shots
  with the FEC withheld test folds, so the scores here measure generalisation to unseen archive
  shots, not a zero-shot result on the FEC folds.
* **Not fully blind OSF.** Some of OSF's shape choices used the validation shots and aggregate
  scores on the public-test fold.
* **Archive input format.** The archive ingest resamples the currents differently from the FEC
  release format (up to 44 kA in $I_p$ on shot 28348, a 0.1 ms clock offset), overlays edge
  Thomson from `aye`, and keeps a 131-entry core radius array.
* **Small validation set.** The three validation shots are consecutive (115 frames); pooled $R^2$
  values on them are noisy, so the archive validation split carries the selection.
* **Non-causal context.** M9 uses EFIT frames up to four steps ahead; it is an offline smoother.
* **No vessel currents** in OSF, and a fixed elliptical current support.
