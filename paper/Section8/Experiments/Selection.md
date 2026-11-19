# Selection runs

Each weight sweep trains member 101 for 10 epochs per setting and scores it on `select30` (flux
only; $q_{95}$ and $\beta_N$ held constant except where the heads are trained, so $R^2_{qb}$ is
comparable only within E3 and E4). The setting with the lowest $\mathcal R$ against the fixed
reference ensemble carries forward (E3: the highest $R^2_{qb}$ among settings within 5% of the
lowest $\mathcal R$).

@import "../../generated/exp_sweeps.md"

Table: Weight sweeps; weights are $(w_b,w_a)$, $(w_b,w_a,w_s)$ and $(w_b,w_a,w_s,w_c,w_t)$; the chosen setting carries forward.

The soft-LCFS term raised $\mathcal R$ at both weights tried, so E2–E4 use the boundary band alone
($w_b=3$). The larger head weight gave both the lower $\mathcal R$ and the higher $R^2_{qb}$; the
smaller critical-point weight gave the lowest $\mathcal R$ of all sweeps.

@import "../../generated/exp_members.md"

Table: Per-member checkpoint selected by $\mathcal R$ on `select30` (E1: best coefficient MSE).

Route-based checkpoint selection chose the best-MSE state for 5 of the 9 route-guided members and
the epoch-44 or epoch-49 state for the other four; it changed little relative to the MSE rule.
