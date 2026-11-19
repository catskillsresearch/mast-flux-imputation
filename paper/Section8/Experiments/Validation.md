# Validation shots and archive validation

@import "../../generated/exp_demo.md"

Table: The three validation shots (115 frames); gap is the share of frames with an LCFS tip gap of at least 50 mm. $\mathcal R$ relative to E1.

@import "../../generated/exp_archval100.md"

Table: `archval100` (100 archive-validation shots, 5,737 frames). $\mathcal R$ relative to E1.

On the validation shots E4 has the highest $S$ (0.867116 against 0.863300 for E1). The flux term
is not the cause: $R^2_\psi$ is 0.998764 for E4 and 0.998826 for E1. The difference is in the
scalars: E1 and E2 take $q_{95}$ and $\beta_N$ from OSF through the E5 maps and reach
$R^2_{q_{95}}=0.635596$ on these three shots; the heads of E3 and E4 reach $R^2_{q_{95}}=0.973$
and 0.975 but $R^2_{\beta_N}$ of only 0.297 and 0.417, against 0.577 for E1. E2–E4 have fewer
tip-gap frames than E1 (22.6–28.7% against 31.3%).

On `archval100` the heads raise $S$ from 0.8589 (E1) to 0.9343 (E3, E4), and E4 has the lowest
$\mathcal R$, 0.9557 relative to E1, which makes E4 the final pick. E2 alone is worse than E1 on
$\mathcal R$ (1.0135).

The E5 maps fitted on `osftrain200` give OSF the following scalar terms:

@import "../../generated/exp_e5.md"

Table: OSF with the E5 maps ($a_q=0.626$, $b_q=1.957$, $a_b=1.013$, $b_b=0.093$).
