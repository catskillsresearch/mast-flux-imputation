# Abstract

A model judged by a composite score can reach a high value of that score with a simple loss, such
as mean squared error on a compressed representation of the field, and still fail to reproduce the
physical observables the score is meant to summarise. We state the hypothesis that such a model
generalises less well than one guided by matching those observables in intensity (the flux field
and its scalar time series) and in geometry (the last closed flux surface, O-point and X-points),
and we formalise a two-part objective: the *target* is the challenge score; the *route* is a set of
observable distances computed frame by frame. The setting is the reconstruction of MAST poloidal
flux $\psi(R,Z)$ on a $65\times65$ grid, with $q_{95}$ and $\beta_N$, from poloidal-field coil
currents, plasma current, toroidal-field current and midplane Thomson scattering, with no magnetic
probe or flux-loop input. The model is a hybrid of M9, a PCA-basis neural surrogate trained on
FAIR-MAST EFIT++ flux, and OSF, a free-boundary Grad–Shafranov forward solver over the 812-element
MAST coil set. Trained with coefficient MSE alone on FAIR-MAST (3,258 shots, 10% of them for validation), the
hybrid reaches
$S=0.874196$ and $R^2_\psi=0.997808$ on a sealed, seed-42 FAIR-MAST holdout disjoint from training
(65 shots, 3,449 diverted frames). Yet it has a top or bottom boundary-tip gap of at least 50 mm in
24.9% of frames, $R^2=0.697184$ for lower triangularity and $R^2=0.444019$ for $q_{95}$, and
between the three validation shots and the holdout its X-point offset, p95 boundary distance and
$q_{95}$ error grow while its score rises. An automated observable audit measures the route for
every frame, and a FAIR-MAST-only training regime compares MSE-only and observable-guided training
on the same data and holdout. Adding the scalar observables to training raises holdout $S$ from
0.874196 to 0.931421 with no use of challenge labels, and training on per-shot curve distances of
the scalar time series raises it to 0.934444, with lower-triangularity $R^2$ 0.761293; the
geometric route terms tried (boundary band, soft LCFS, critical points) do not improve the
boundary beyond coefficient MSE.
