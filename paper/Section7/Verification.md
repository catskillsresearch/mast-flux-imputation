# Verification of the extraction

| check | result |
|---|---|
| ported OSF vs original code, 3 demo shots | $\max|\Delta\psi|$, $\max|\Delta q_{95}|$, $\max|\Delta\beta_N|$ = 0 |
| ported M9 inference vs original, 3 demo shots | $\max|\Delta\psi|$ = 0 |
| `pipeline.score` vs the source repository's scorer, same predictions | 19 of 19 result fields identical, all four runs |
| M9 split / QC / $|I_p|$ window recomputed | keep list identical; $I_p^{\rm lo}=328.7293519015368$ kA |
| shipped M9 checkpoints | sha256 match the source `SHA256SUMS` |
| FAIR-MAST labels vs FEC demo rows | $\psi$ bit-identical; $q_{95}$, $\beta_N$ and the diverted-frame counts (36/42/37) exact |
| tables of Section 8, regenerated from the saved score files | identical to the repository README |

Not verified: a full M9 retrain to matching weights (only a one-epoch smoke test was run; GPU
non-determinism means a retrain reproduces the recipe, not the bytes), and predictions on the
public-test fold.
