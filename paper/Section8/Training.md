# Using the audit in training

The audit turns inspection into training signal:

1. **Reweighting and targeted sampling** of frames and shots in the over-represented failure
   classes, in place of uniform shot-balanced sampling;
2. **Route terms in the loss** beyond MSE on PCA coefficients: LCFS-distance, X-point and
   $\kappa$/$\delta$ penalties evaluated through differentiable surrogates of the quantities in
   Eq. \eqref{eq:route-geometry}, which would also be the first physics-informed element of M9;
3. **A checkpoint gate** defined by route quantities and failure-class rates on a fixed archive
   validation set together with the three validation shots, replacing the current acceptance
   threshold on the three shots alone ($R^2_\psi\ge0.9976$).

The thesis test is a paired comparison: arm A trains with coefficient MSE and selects by
$R^2_\psi$, arm B adds items 1–3; both use the same FAIR-MAST training shots, seeds and network,
and both are scored once on the same sealed holdout with the audit. H predicts that B matches or
exceeds A in $S$ on the holdout and reduces the route quantities, with a smaller gap between
validation and holdout.
