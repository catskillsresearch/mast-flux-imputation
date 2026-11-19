# Target and route

The quantity a user of an imputed equilibrium asks for is a score: here the composite $S$, which
pools $R^2$ of $\psi$, $q_{95}$ and $\beta_N$ with a boundary distance and seven shape-consistency
terms (Metric section). Call this the **target**. $R^2_\psi$ carries 55% of the weight and is pooled
over pixels, so it is dominated by the large-scale vacuum field that the coil currents largely
determine; it says little about where the plasma boundary is.

The **route** is what a physicist checks for each shot in the viewer of the repository
(`app/fec_site`): the time series of $q_{95}$, $\beta_N$, elongation, triangularity, volume, the
magnetic axis and $\ell_i$, and, frame by frame, the last closed flux surface (LCFS), the O-point
and the X-points. The model is calibrated on the route directly: its loss contains the per-shot
curve distance between the predicted and the EFIT++ trace of each scalar, and its checkpoints are
chosen by the same distances on archive validation shots. The results report both the target and
the route.
