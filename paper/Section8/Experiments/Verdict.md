# Verdict

H predicted that the route-guided pick would match or exceed the MSE-only E1 on holdout $S$ and on
the route metrics.

* *Target.* Confirmed for $S$: 0.931421 against 0.874196. The whole gain comes from the scalar
  terms, through the network's scalar heads trained on EFIT++ $q_{95}$, $\beta_N$, $R_{\rm axis}$,
  $Z_{\rm axis}$, $\ell_i$, which is the intensity part of the route (scalar time series). Flux
  $R^2_\psi$ is unchanged.
* *Geometry.* Not confirmed. The boundary band and soft-LCFS terms (E2) did not improve on MSE in
  $\mathcal R$; the critical-point and temporal terms (E4) reduced the tip-gap rate and the
  Hausdorff distances on the holdout but increased the O- and X-point offsets and the flip rate,
  so the composite $\mathcal R$ of E4 relative to E1 is 1.0186 on the holdout, after 0.9557 on
  `archval100`. Route-based checkpoint selection mostly chose the best-MSE state.
* *Generalisation.* The MSE-only flux generalises well: E1's $R^2_\psi$ is 0.998826 on the
  validation shots and 0.997808 on the holdout. FAIR-MAST-trained scalars (E5 maps, heads) carry
  over to unseen shots; the boundary errors grow from validation shots to holdout for every model.

The supported form of the thesis is therefore narrower than H: an MSE fit of the flux leaves the
score limited by the quantities MSE does not see, here $q_{95}$ and $\beta_N$, and adding those
observables to training raises the score on unseen shots; the geometric route terms as
implemented here (pixel band, soft LCFS, gradient at critical points) do not yet improve the
boundary beyond what coefficient MSE achieves.
