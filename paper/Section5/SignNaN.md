# Sign and missing-value policy

A single fold-global sign $s\in\{+1,-1\}$ is chosen to minimise $\mathrm{SS}_{\rm res}(\psi)$, and the
contour work uses the axis sign $\sigma_{\rm ax}s$; all runs here select $s=+1$. Non-finite
predicted pixels or scalars are replaced by the ground-truth mean, and a derived scalar that cannot
be computed from the prediction is replaced in the same way, so failures cost score rather than
being skipped.
