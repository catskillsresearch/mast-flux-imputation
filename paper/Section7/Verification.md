# Verification

| check | result |
|---|---|
| `pipeline.score` vs an independent implementation of the scorer reduction, same predictions | 19 of 19 result fields identical, all four runs checked |
| `pipeline.audit` $S$ vs `pipeline.score` $S$ | identical on every audited run |
| sealed split recomputed, disjointness checks | all true (Section 2.4) |
| FAIR-MAST labels vs FEC demo rows | $\psi$ bit-identical; $q_{95}$, $\beta_N$ and the diverted-frame counts (36/42/37) exact |
| result tables, regenerated from the saved audit summaries | identical to the tables in this paper and the repository README |

Not verified: predictions on the public-test fold, which has no labels. GPU non-determinism means a
retrain reproduces the recipe, not the bytes, of a checkpoint.
