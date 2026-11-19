# Loss weight, seeds and selection

**Curve-distance weight.** $w_{cd}$ was chosen on seed 101 with 40-epoch runs by the lowest $J$:

@import "../generated/model_sweep.md"

Table: Curve-distance weight against the best validation objective $J$.

**Seeds.** Five members were then trained with $w_{cd}=0.01$:

@import "../generated/model_training.md"

Table: Training records of the five members. "Returns to best weights" counts the spike guard.

**Selection.** For $k=1,2,3$ the ensemble of the five members' $k$-th best EMA states (by $J$) is
formed. The first $k$ whose archive-validation score $S$ on `archval100` reaches the fixed
threshold 0.934308 is used:

@import "../generated/model_selection.md"

Table: Ensemble selection on archive validation.

The rank-1 states fail the threshold; the rank-2 states (epochs 103, 154, 106, 121 and 86 for seeds
101–505) pass and form the model.
