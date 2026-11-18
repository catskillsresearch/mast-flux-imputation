# Limitations

* **Labels are magnetics-based.** EFIT++ used probes and flux loops; the task is imputation of a
  magnetics-constrained reconstruction, and its errors are inherited.
* **Not comparable with challenge leaderboards.** The FAIR-MAST training pool very likely shares
  shots with the challenge's withheld test sets, so the scores here measure archive
  generalisation, not a zero-shot challenge result.
* **Not fully blind OSF.** OSF's global factors and some shape choices used aggregate scores on an
  external MAST test set and the validation shots.
* **Archive input format.** The archive ingest resamples the currents differently from the FEC
  release format (up to 44 kA in $I_p$ on shot 28348, a 0.1 ms clock offset), overlays edge
  Thomson from `aye`, and keeps a 131-entry core radius array.
* **Weak scalars.** $q_{95}$ and $\beta_N$ come from OSF's simplified read-outs (vacuum $F$, midplane
  pressure); M9's scalar heads are untrained.
* **Non-causal context.** M9 uses EFIT frames up to four steps ahead; it is an offline smoother.
* **No vessel currents** in OSF, and a fixed elliptical current support.
* **Retrain not reproduced** end to end; see Section 9.1.
