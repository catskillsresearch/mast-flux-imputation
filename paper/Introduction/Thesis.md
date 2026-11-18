# Thesis: target and route

The quantity a user of an imputed equilibrium asks for is a score: here the Fusion Equilibrium
Challenge composite $S$, which pools $R^2$ of $\psi$, $q_{95}$ and $\beta_N$ with a boundary
distance and shape consistency terms (Section 6). Call this the **target**. The cheapest way to
raise it is to fit the dominant term with the simplest loss, for example mean squared error on the
leading principal components of $\psi$, since $R^2_\psi$ carries 55% of the weight.

Our thesis is that such a fit can give a superficially high target score on the data it was fitted
or selected on, but need not generalise, unless training and model selection are guided by the
**route**: matching the physical observables a physicist checks in a viewer, in *intensity*
(the $\psi$ field and the scalar time series $q_{95}$, $\beta_N$, $\kappa$, $\delta$, $\ell_i$, $V$)
and in *geometry* (the last closed flux surface, the O-point and the X-points). A pooled pixel
$R^2$ near one is dominated by the large-scale vacuum field; it says little about where the
boundary is, which is what the observables measure.

We treat this as a hypothesis. Section 7 defines the objective, and Section 7.5 reports the
evidence available from the existing models, which is consistent with it: the hybrid model's
$R^2_\psi$ is above 0.997 on both the validation shots and a sealed holdout, while its boundary-tip
and $q_{95}$ errors are large and grow on the holdout, and lower triangularity is poorly reproduced
on both. The direct test, training the
same network on the same FAIR-MAST data with and without observable terms and scoring both once on
the same sealed holdout, is planned (Section 10).
