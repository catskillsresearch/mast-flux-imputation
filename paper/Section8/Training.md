# Using the audit in training

The audit turns inspection into training signal:

1. **Reweighting and targeted sampling** of frames and shots in the over-represented failure
   classes, in place of uniform shot-balanced sampling;
2. **Shape-aware loss terms** beyond MSE on PCA coefficients — for example LCFS-distance, X-point
   and $\kappa$/$\delta$ penalties evaluated through differentiable surrogates, which would also be
   the first physics-informed element of M9;
3. **A checkpoint gate** defined by failure-class rates on a fixed archive validation set, replacing
   the current acceptance threshold on the three demo shots ($R^2_\psi\ge0.9976$), which is both
   small and not held out.
