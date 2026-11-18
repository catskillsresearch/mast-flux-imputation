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
probe or flux-loop input. Two models minimise different parts of the objective: OSF, a
free-boundary Grad–Shafranov forward solver over the 812-element MAST coil set, and M9, a PCA-basis
neural surrogate (not physics-informed) trained with coefficient MSE on 3,622 FAIR-MAST archive
shots. On a sealed, seed-42 FAIR-MAST holdout disjoint from M9 training (65 shots, 3,449 diverted
frames), OSF reaches $S=0.618634$ and the hybrid M9 + OSF $S=0.862677$ with
$R^2_\psi=0.997771$. Yet the same hybrid has a top or bottom boundary-tip gap of at least 50 mm in
24.2% of frames, $R^2=0.577001$ for lower triangularity and $R^2=0.342298$ for $q_{95}$; between
the three validation shots used for model selection and the holdout its flux score barely moves
while its boundary and $q_{95}$ errors grow. These measurements are consistent with the hypothesis but do not test
it; we describe an automated observable audit, implemented, and a FAIR-MAST-only training regime,
planned, that compares MSE-only and observable-guided training on the same data and holdout.
