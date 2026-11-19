# Ensemble

Five members differ only in their initialisation seed $s\in\{101,202,303,404,505\}$; the data, split,
basis and feature statistics are shared. The flux prediction is the mean of the reconstructed maps
and the scalar prediction the mean of the heads,

$$
\hat\psi = \frac15\sum_{s}\Big(\mu+C^{\top}f_s(\tilde x)\Big),\qquad
\hat u_j = \frac15\sum_s \hat u_{j,s}.
$$

Which training state of each member enters the ensemble is set by the selection rule of the
Calibration section.
