# Evidence from the MSE-only model

The audit was run on the hybrid trained with coefficient MSE alone (experiment E1 of Section 10.6:
M9 flux, OSF $q_{95}$ and $\beta_N$ through read-out maps fitted on FAIR-MAST) and on OSF alone
with the same read-out, on the three validation shots (115 frames) and on the sealed holdout (65
shots, 3,449 frames; never used for fitting or selection). It reproduces the official scorer's $S$
on every run.

@import "../generated/audit_target.md"

Table: Target terms from the audit (scorer v3.1.0 definitions).

@import "../generated/audit_route.md"

Table: Route quantities from the audit: per-frame means or rates pooled over all frames; the lag is averaged over shots and scalar traces.

Three observations bear on the hypothesis.

1. **A near-saturated flux score coexists with large observable errors.** On the holdout the hybrid
   has $R^2_\psi=0.997808$, yet a boundary-tip gap of at least 50 mm in 24.9% of frames, a p95
   Hausdorff distance of 104.2 mm, a limited/diverted flip in 2.6% of frames,
   $R^2(\delta_{\rm bot})=0.697184$ and $R^2_{q_{95}}=0.444019$. The term that coefficient MSE
   targets is almost perfect; the boundary and the scalars are not.
2. **From validation shots to holdout, the score rises while much of the route degrades.**
   $R^2_\psi$ goes from 0.998826 to 0.997808 and $S$ from 0.863300 to 0.874196, while the p95
   Hausdorff distance goes from 87.5 to 104.2 mm, the mean X-point offset from 8.9 to 20.5 mm, the
   flip rate from 0.0% to 2.6%, $R^2_{q_{95}}$ from 0.635596 to 0.444019 and $R^2(\ell_i)$ from
   0.923484 to 0.664240. The tip-gap rate falls (31.3% to 24.9%). A rising score does not mean the
   observables carried over.
3. **The physics model shows the boundary pattern more strongly.** OSF goes from $S=0.750287$ to
   $0.640558$, and its tip-gap rate from 73.0% to 90.5%.

These observations are consistent with H but do not test it. The validation shots are three
consecutive shots, so their pooled $R^2$ values are noisy: $R^2(Z_{\rm axis})=-89.624564$ there
reflects a tiny between-shot variance, and several consistency terms are *higher* on the holdout
(Cons 0.640195 versus 0.841297). The two populations also differ in shot mix. The direct test is
the FAIR-MAST-only regime of Section 10.6: the same network, data, seeds and holdout, trained once
with coefficient MSE alone and once with route terms in the loss and in checkpoint selection,
compared term by term with this audit.
