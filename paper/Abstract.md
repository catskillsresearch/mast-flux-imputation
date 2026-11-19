# Abstract

We describe a model that imputes the MAST poloidal flux map $\psi(R,Z)$ on a $65\times65$ grid,
together with $q_{95}$ and $\beta_N$, at every EFIT++ time from measurements that exclude magnetic
probes and flux loops: the poloidal-field and toroidal-field coil currents, the plasma current and
the core Thomson-scattering profiles. The model is trained only on open data from the UK FAIR-MAST
archive, with the EFIT++ reconstructions stored there as ground truth. It is an ensemble of five
residual networks that output the coefficients of a 60-mode principal-component basis of EFIT++
flux, with linear scalar heads for $q_{95}$ and $\beta_N$. Calibration combines a coefficient loss
with per-shot curve distances between the predicted and EFIT++ traces of $q_{95}$, $\beta_N$,
elongation, triangularity, plasma volume and magnetic-axis position; the shape traces are read from
the predicted flux by differentiable extractors, and checkpoints are selected on archive validation
shots only. Performance is reported as ground truth against prediction with the composite score
$S$ of the FEC MAST scoring code. On a sealed holdout of 65 archive shots (3,449 frames) the model
reaches $S=0.934444$, with $R^2_\psi=0.997832$, $R^2_{q_{95}}=0.913296$ and $R^2_{\beta_N}=0.677889$;
on three held-out MAST validation shots downloaded from FAIR-MAST (115 frames) it reaches
$S=0.899573$. The largest remaining errors are $\beta_N$ in the last frames of each shot, the
plasma boundary near its top and bottom, and the shape scalars of the validation shots.
