# Verification

| check | result |
|---|---|
| `pipeline.score` vs an independent implementation of the scorer, same predictions | $S$ and every term identical on the holdout and on the validation shots |
| `pipeline.audit` $S$ vs `pipeline.score` $S$ | identical on both folds |
| sealed split recomputed, disjointness checks | all true |
| validation-shot diverted-frame counts from FAIR-MAST `efm` X-points | 36 / 42 / 37 |
| result tables, regenerated from the saved prediction files and audit summaries | identical to the tables in this paper |

GPU non-determinism means a retrain reproduces the recipe, not the bytes, of a checkpoint.
