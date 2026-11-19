# Validation shots: score box

The three labelled FEC shots 28348, 28350 and 28351 (115 frames) are held-out validation shots for
the trained models: they are excluded from training, the PCA basis and checkpoint selection. They
are scored once for E6, against EFIT++.

@import "../generated/e6_scorebox_demo.md"

Table: E6 on the three validation shots: composite score, its terms and the seven consistency components (pooled $R^2$).

On these shots E6 reaches $S=0.899567$ with $R^2_\psi=0.998879$, $R^2_{q_{95}}=0.976192$ and
$R^2_{\beta_N}=0.663239$. With only three consecutive shots, the pooled $R^2$ of a scalar with
little between-shot variance can be extreme: $R^2(Z_{\rm axis})=-5.467374$ comes from a tiny
$\mathrm{SS}_{\rm tot}$ (the per-shot curve distances of the next note are a few millimetres), not
from a large error. The weakest physical terms are $\delta_{\rm bot}$ (0.427796) and $\kappa$
(0.827419).
