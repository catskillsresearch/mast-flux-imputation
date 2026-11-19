# Evidence from the existing models

The audit was run on the saved predictions of both shipped models, on the three validation shots
(115 frames; used for OSF tuning and the M9 acceptance threshold, Sections 4.5 and 5.3) and on the sealed
holdout (65 shots, 3,449 frames; never used for fitting or selection). It reproduces the official
scorer's $S$ on every run.

@import "../generated/audit_target.md"

Table: Target terms from the audit (scorer v3.1.0 definitions).

@import "../generated/audit_route.md"

Table: Route quantities from the audit: per-frame means or rates pooled over all frames; the lag is averaged over shots and scalar traces.

Three observations bear on the hypothesis.

1. **A near-saturated flux score coexists with large observable errors.** On the holdout the hybrid
   has $R^2_\psi=0.997771$, yet a boundary-tip gap of at least 50 mm in 24.2% of frames, a p95
   Hausdorff distance of 108.6 mm, a limited/diverted flip in 2.9% of frames,
   $R^2(\delta_{\rm bot})=0.577001$ and $R^2_{q_{95}}=0.342298$. The term that coefficient MSE
   targets is almost perfect; the boundary and the scalars are not.
2. **From selection shots to holdout, the target barely moves while the route degrades.** For the
   hybrid, $R^2_\psi$ goes from 0.998555 to 0.997771 and $S$ from 0.871921 to 0.862677, while the
   p95 Hausdorff distance goes from 64.8 to 108.6 mm, the mean X-point offset from 7.9 to 21.2 mm,
   the tip-gap rate from 20.0% to 24.2%, $R^2_{q_{95}}$ from 0.818390 to 0.342298 and
   $R^2(\ell_i)$ from 0.933054 to 0.668520. Selection by $R^2_\psi$ on three shots did not carry
   over to the observables.
3. **The physics model tuned by eye shows the same pattern more strongly.** OSF goes from
   $S=0.774217$ to $0.618634$, and its tip-gap rate from 73.0% to 90.5%.

These observations are consistent with H but do not test it. The validation shots are three
consecutive shots, so their pooled $R^2$ values are noisy: $R^2(Z_{\rm axis})=-60.084051$ there
reflects a tiny between-shot variance, and several consistency terms are *higher* on the holdout
(Cons 0.606137 versus 0.820703). The two populations also differ in shot mix. The shipped M9's training
logs contain no route quantities. The direct test is the FAIR-MAST-only regime of Section 10.6: the same network, data, seeds and holdout, trained once with
coefficient MSE alone and once with route terms in the loss and in checkpoint selection, compared
term by term with this audit.
