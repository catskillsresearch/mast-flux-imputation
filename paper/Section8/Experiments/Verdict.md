# Verdict

**Acceptance against the shipped calibration.** Under the same scorer, the FAIR-MAST-only pick E4
scores higher than the shipped hybrid on the sealed holdout ($S=0.931421$ against $0.862677$) and
lower on the three validation shots ($0.867116$ against $0.871921$). The shipped hybrid's scalar
read-out was tuned on those three shots and on leaderboard probes; E4 was tuned on neither. By the
strict rule of `docs/TRAINING_REGIME.md` (at least as good on every metric on the validation shots)
E4 is not accepted; by holdout $S$, the quantity the challenge measures, it is the better model.

**The hypothesis.** H predicted that the route-guided pick would match or exceed the MSE-only E1 on
holdout $S$ and on the route metrics.

* *Target.* Confirmed for $S$: 0.931421 against 0.874196. The whole gain comes from the scalar
  terms, through the network's scalar heads trained on EFIT++ $q_{95}$, $\beta_N$, $R_{\rm axis}$,
  $Z_{\rm axis}$, $\ell_i$, which is the intensity part of the route (scalar time series). Flux
  $R^2_\psi$ is unchanged.
* *Geometry.* Not confirmed. The boundary band and soft-LCFS terms (E2) did not improve on MSE on
  any split; the critical-point and temporal terms (E4) reduced the tip-gap rate and the Hausdorff
  distances on the holdout but increased the O- and X-point offsets and the flip rate, and the
  composite $\mathcal R$ of E4 (0.9906) is slightly worse than that of E1 (0.9742). Route-based
  checkpoint selection mostly chose the best-MSE state.
* *Generalisation.* The MSE-only flux generalises well: E1's $R^2_\psi$ on the holdout equals the
  shipped model's and its $\mathcal R$ is below 1. The degradation from validation shots to
  holdout seen for the shipped hybrid in Section 7.5 is in its scalar read-out, and FAIR-MAST-only
  scalars (E5 maps, heads) remove it.

The supported form of the thesis is therefore narrower than H: an MSE fit of the flux leaves the
customer's score limited by the quantities MSE does not see, here $q_{95}$ and $\beta_N$, and
adding those observables to training raises the score on unseen shots; the geometric route terms
as implemented here (pixel band, soft LCFS, gradient at critical points) do not yet improve the
boundary beyond what coefficient MSE achieves.
