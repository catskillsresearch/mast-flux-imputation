# Model

The model maps the measured inputs of each EFIT frame to 60 principal-component coefficients of
the EFIT++ flux and to five scalars. Five networks of identical architecture, differing only in
their initialisation seed, are averaged.

![The model: per-frame inputs are augmented in time, passed through a present-frame and a context residual encoder joined by a zero-initialised gate, and mapped linearly to 60 PCA coefficients, reconstructed on the $65\times65$ grid, and to five scalar heads; five seeds are averaged. Training adds per-shot curve distances computed through soft read-outs of the predicted flux.](figures/diag_model_arch.png){#fig:arch}

- [PCA basis of EFIT++ flux](Section3/PCABasis.md)
- [Features and temporal context](Section3/Features.md)
- [Network and scalar heads](Section3/Network.md)
- [Differentiable shape read-outs](Section3/Readouts.md)
- [Ensemble](Section3/Ensemble.md)
