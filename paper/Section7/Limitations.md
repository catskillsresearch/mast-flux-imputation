# Limitations

* **Labels are magnetics-based.** EFIT++ used probes and flux loops; the task is imputation of a
  magnetics-constrained reconstruction, and its errors are inherited.
* **Archive overlap.** Of roughly 4,566 archive shots with EFIT and core Thomson, FEC curated 2,416;
  the FAIR-MAST training pool very likely contains FEC hidden-fold shots, so M9 is not a zero-shot
  MAST result in the challenge's sense.
* **Not fully blind OSF.** OSF's global factors and some shape choices used aggregate public-test
  scores and the demo shots.
* **Holdout inputs differ from FEC rows.** The archive ingest resamples the currents differently
  (up to 44 kA in $I_p$ on shot 28348, a 0.1 ms clock offset), overlays edge Thomson from `aye`,
  and keeps a 131-entry core radius array. Scores on true FEC rows may differ.
* **Weak scalars.** $q_{95}$ and $\beta_N$ come from OSF's simplified read-outs (vacuum $F$, midplane
  pressure); M9's scalar heads are untrained.
* **Non-causal context.** M9 uses EFIT frames up to four steps ahead; it is an offline smoother.
* **No vessel currents** in OSF, and a fixed elliptical current support.
* **Retrain not reproduced** end to end; see Section 9.1.
