# Sealed holdout and disjointness

The holdout is the split built into the M9 training code. Let $\mathcal K$ be the sorted list of
3,738 ingested keys and $\mathcal D$ the three demo shots. Then

$$
\mathcal S = \mathrm{first}_{80}\big(\mathrm{shuffle}_{42}(\mathcal K\setminus\mathcal D)\big),\qquad
\mathcal P = (\mathcal K\setminus\mathcal D)\setminus\mathcal S,\quad |\mathcal P| = 3655,
$$

with `random.Random(42).shuffle`. Quality control on $\mathcal P$ keeps 3,622 shots
($\mathcal Q$) and quarantines 33. `data/fair_mast/make_holdout.py` recomputes the split and checks

| check | result |
|---|---|
| recomputed $\mathcal P$ equals the QC training set (keep ∪ quarantine) | true (3,655) |
| QC keep list equals the shipped `mast_qc_keep.txt` | true (3,622) |
| $\mathcal Q\cap\mathcal S$, $\mathcal P\cap\mathcal S$, $\mathcal D\cap\mathcal S$ | ∅, ∅, ∅ |
| quarantined sealed shot | 27373 ∈ $\mathcal S$ |

and every M9 checkpoint records `seed = 42`, `train_shots = 3622`. The holdout is therefore
disjoint from all M9 training and validation data.

**FEC-protocol cut.** Each sealed shot is restricted to its diverted EFIT slices with the FEC rule;
inputs keep their full clocks. Fifteen sealed shots have no diverted slice (24901, 26506, 29932,
24876, 28295, 27681, 23248, 27621, 27720, 29297, 26273, 27819, 27778, 23424, 30442) and are
excluded, leaving **65 shots and 3,449 frames** out of 5,280 EFIT slices. Edge Thomson missing from
the ingest is overlaid from `aye` (60 of 65 shots).

**Caveat.** OSF was not fitted to these shots' labels, but its global scalar factors were tuned
against aggregate leaderboard scores on the public-test fold, and the FAIR-MAST pool likely
overlaps the FEC hidden folds (Section 8).
