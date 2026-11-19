# Conclusion

A score built from pooled $R^2$ values can be pushed high by a simple loss: the hybrid of a
PCA-basis neural surrogate trained with coefficient MSE on FAIR-MAST and a forward Grad–Shafranov
solver reaches $R^2_\psi=0.997808$ and $S=0.874196$ on a sealed archive holdout from coil currents,
plasma current and Thomson scattering alone, with no magnetic sensors. The observables tell a less
complete story: a 50 mm or larger boundary-tip gap in 24.9% of holdout frames,
$R^2_{q_{95}}=0.444019$, and X-point, boundary and $q_{95}$ errors that grow from the three
validation shots to the holdout while the score rises. We have therefore proposed that the
objective for flux imputation should combine the score as target with observable distances in
intensity and geometry as route, implemented an audit that measures the route for every frame, and
run a FAIR-MAST-only experiment that tests whether route-guided training generalises better than
MSE alone. The intensity route carries the result. Trained on EFIT++ $q_{95}$, $\beta_N$, axis and
$\ell_i$, the network reaches $S=0.931421$ on the sealed holdout (E4); trained in addition on the
curve distance of each shot's time series, with weight on the end of the shot and on
triangularity, it reaches $S=0.934444$ (E6), with Consistency 0.862171 and lower-triangularity
$R^2$ 0.761293, while its flux $R^2_\psi$ stays at 0.997832. On the three validation shots E6
scores $S=0.899567$. The geometric route is not yet solved: E6 pays for its shape scalars with
more boundary-tip gaps and larger X-point offsets than the MSE-only model, and most of its
remaining $\beta_N$ error sits in the last frames of each shot. Differentiable boundary and X-point terms that
agree with the scorer's extractor are the next step. The repository, the paper sources and all
result tables are reproducible from public data, and the data each component was trained or tuned
on is documented in Section 9.2.
