# Conclusion

From coil currents, plasma current and core Thomson scattering alone, with no magnetic sensors,
the model imputes the MAST flux map with $R^2_\psi=0.997832$ and reaches a composite score
$S=0.934444$ on a sealed holdout of 65 FAIR-MAST shots, and $S=0.899573$ on three held-out
validation shots. It is a 60-mode principal-component basis of EFIT++ flux driven by an ensemble of
five residual networks with scalar heads for $q_{95}$ and $\beta_N$, calibrated on per-shot curve
distances of the scalar time series read through differentiable extractors, and selected on
archive validation shots. Training, selection and evaluation use only open FAIR-MAST data; the
scoring code and output template are those of the FEC MAST task. The remaining errors are
$\beta_N$ at the end of each shot, the top and bottom of the plasma boundary and the X-point
position, and the shape scalars of the validation shots. The repository, the paper sources and all
result tables are reproducible from public data.
