# Sealed holdout and disjointness

The holdout is the split built into the M9 training code. Let $\mathcal K$ be the sorted list of
3,738 ingested keys and $\mathcal D$ the three demo shots. Then

$$
\mathcal S = \mathrm{first}_{80}\big(\mathrm{shuffle}_{42}(\mathcal K\setminus\mathcal D)\big),\qquad
\mathcal P = (\mathcal K\setminus\mathcal D)\setminus\mathcal S,\quad |\mathcal P| = 3655,
$$

with `random.Random(42).shuffle`. Quality control on $\mathcal P$ keeps the set $\mathcal Q$
(Section 4). `data/fair_mast/make_holdout.py` recomputes the split and checks

| check | result |
|---|---|
| recomputed $\mathcal P$ equals the QC training set (keep ∪ quarantine) | true (3,655) |
| $\mathcal Q\cap\mathcal S$, $\mathcal P\cap\mathcal S$, $\mathcal D\cap\mathcal S$ | ∅, ∅, ∅ |
| quarantined sealed shot | 27373 ∈ $\mathcal S$ |

and every M9 training run uses the same seed-42 split. The holdout is therefore disjoint from all
M9 training and validation data.

**FEC-protocol cut.** Each sealed shot is restricted to its diverted EFIT slices with the FEC rule;
inputs keep their full clocks. Fifteen sealed shots have no diverted slice (24901, 26506, 29932,
24876, 28295, 27681, 23248, 27621, 27720, 29297, 26273, 27819, 27778, 23424, 30442) and are
excluded, leaving **65 shots and 3,449 frames** out of 5,280 EFIT slices. Edge Thomson missing from
the ingest is overlaid from `aye` (60 of 65 shots).

Fourteen of the fifteen excluded shots are no-plasma shots (Section 2.2); the fifteenth is 24876.

**Caveat.** OSF was not fitted to these shots' labels, but some of its hand-set values were chosen
with reference to aggregate scores on the FEC public-test fold (Section 5.3), which may share shots
with the archive.
