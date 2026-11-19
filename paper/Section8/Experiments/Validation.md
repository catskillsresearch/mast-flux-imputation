# Validation shots and archive validation

@import "../../generated/exp_demo.md"

Table: The three validation shots (115 frames). Shipped is the M9 + OSF hybrid; gap is the share of frames with an LCFS tip gap of at least 50 mm. $\mathcal R$ relative to the shipped hybrid.

@import "../../generated/exp_archval100.md"

Table: `archval100` (100 archive-validation shots, 5,737 frames). $\mathcal R$ relative to the shipped hybrid.

On the validation shots no experiment reaches the shipped hybrid's $S=0.871921$; the closest is E4
with $S=0.867116$. The flux term is not the cause: E1 has a higher $R^2_\psi$ (0.998826 against
0.998555). The difference is in the scalars and the boundary. E1 and E2 take $q_{95}$ and $\beta_N$
from OSF through the E5 maps, which fit FAIR-MAST labels on 200 training shots and give
$R^2_{q_{95}}=0.635596$ on these three shots, against 0.818390 with $c^*$, $s^*$; the heads of E3
and E4 reach $R^2_{q_{95}}=0.973$ and 0.975 but $R^2_{\beta_N}$ of only 0.297 and 0.417. Every experiment
has more tip-gap frames than the shipped hybrid (22.6–31.3% against 20.0%).

`archval100` is not a fair comparison with the shipped model: reproducing the shipped members'
validation split (`scripts/shipped_val_overlap.py`, which recovers the shipped 3,265 shots and
211,935 frames) shows that 94 of its 100 shots were training shots of the shipped M9, so the
shipped flux values on it are in-sample. The comparison among E1–E4, which all exclude these
shots, is fair. There the heads raise $S$ from 0.8589 (E1) to 0.9343 (E3, E4), and E4 has the
lowest $\mathcal R$, 1.3371 against 1.3941 for E1, which makes E4 the final pick.

To separate flux from scalars on the validation shots, the experiment flux ensembles were also
combined with the shipped OSF read-out ($c^*$, $s^*$). This diagnostic was not used for any choice,
and $c^*$, $s^*$ are not FAIR-MAST-only.

@import "../../generated/exp_demo_shipscal.md"

Table: Diagnostic: experiment flux with the shipped scalar read-out on the validation shots.

With identical scalars, the MSE-only flux of E1 scores $S=0.878324$, above the shipped hybrid, and
every route-guided flux scores below E1.

The E5 maps alone improve OSF on archive data and lose on the three validation shots, where
$c^*$, $s^*$ were tuned:

@import "../../generated/exp_e5.md"

Table: OSF with the shipped read-out and with the E5 maps ($a_q=0.626$, $b_q=1.957$, $a_b=1.013$, $b_b=0.093$).
