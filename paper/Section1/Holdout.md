# Sealed holdout and disjointness

Let $\mathcal K$ be the sorted list of 3,738 ingested keys and $\mathcal V$ the three validation
shots. Then

$$
\mathcal S = \mathrm{first}_{80}\big(\mathrm{shuffle}_{42}(\mathcal K\setminus\mathcal V)\big),\qquad
\mathcal P = (\mathcal K\setminus\mathcal V)\setminus\mathcal S,\quad |\mathcal P| = 3655,
$$

with `random.Random(42).shuffle`. `data/fair_mast/make_holdout.py` recomputes the split and checks

| check | result |
|---|---|
| recomputed $\mathcal P$ equals the quality-control pool (keep ∪ quarantine) | true (3,655) |
| $\mathcal Q\cap\mathcal S$, $\mathcal P\cap\mathcal S$, $\mathcal V\cap\mathcal S$ | ∅, ∅, ∅ |
| quarantined sealed shot | 27373 ∈ $\mathcal S$ |

where $\mathcal Q$ is the set kept by quality control. The holdout is disjoint from all training
and validation data and is scored once.

**Scored frames.** Each sealed shot is restricted to its diverted EFIT slices; inputs keep their
full clocks. Fifteen sealed shots have no diverted slice (24901, 26506, 29932, 24876, 28295, 27681,
23248, 27621, 27720, 29297, 26273, 27819, 27778, 23424, 30442) and are excluded, leaving **65 shots
and 3,449 frames** out of 5,280 EFIT slices. Fourteen of the fifteen are no-plasma shots; the
fifteenth is 24876.
