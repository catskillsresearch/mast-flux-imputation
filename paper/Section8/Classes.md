# Failure classes and ranked reports

Each frame receives zero or more tags from a small fixed vocabulary — for example *X-point lost*,
*LCFS too wide*, *LCFS too narrow*, *$\kappa$ biased*, *axis displaced*, *temporal jitter*, *lag*,
*drift*, *outside PCA subspace*. Tags are aggregated into ranked worst-shot and worst-frame tables
whose rows link directly to the corresponding frame in the viewer, so that inspection by eye is spent
on the cases the audit has already identified.
