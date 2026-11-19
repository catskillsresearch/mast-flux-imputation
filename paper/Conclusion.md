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
run a FAIR-MAST-only experiment that tests whether route-guided training generalises better than
MSE alone. The intensity route, the scalar time series, carries the result: trained on EFIT++
$q_{95}$, $\beta_N$, axis and $\ell_i$, the network reaches $S=0.931421$ on the sealed holdout
against 0.874196 for MSE only and 0.862677 for the shipped hybrid, while its flux $R^2_\psi$ is
unchanged. On the three validation shots it scores 0.867116 against 0.871921 for the shipped
hybrid, whose read-out was tuned there. The geometric route terms tried here do not yet improve
the boundary beyond coefficient MSE; better differentiable boundary terms are the next step. The repository, the paper sources and all result tables are reproducible from
public data, and the data each component was trained or tuned on is documented in Section 9.2.
