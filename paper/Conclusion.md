# Conclusion

A score built from pooled $R^2$ values can be pushed high by a simple loss: a PCA-basis neural
surrogate trained with coefficient MSE on 3,622 FAIR-MAST shots, combined with a forward
Grad–Shafranov solver, reaches $R^2_\psi=0.997771$ and $S=0.862677$ on a sealed archive holdout
from coil currents, plasma current and Thomson scattering alone, with no magnetic sensors. The
observables tell a less complete story: a 50 mm or larger boundary-tip gap in 24.2% of holdout
frames, $R^2_{q_{95}}=0.342298$, and boundary and $q_{95}$ errors that grow from the three
selection shots to the holdout while the flux score barely moves. We have therefore proposed that
the objective for flux imputation should combine the score as target with observable distances in
intensity and geometry as route, implemented an audit that measures the route for every frame, and
designed a FAIR-MAST-only experiment that tests whether route-guided training generalises better
than MSE alone. The repository, the paper sources and all result tables are reproducible from
public data, and the data each component was trained or tuned on is documented in Section 9.2.
