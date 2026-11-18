# Ensemble

Three members differ only in their initialisation seed $s\in\{101,202,303\}$ (the data split is
shared). The prediction is the mean of the reconstructed maps,

$$
\hat\psi_{\rm M9} = \frac13\sum_{s\in\{101,202,303\}}\Big(\mu+C^{\top}f_s(\tilde x)\Big).
$$

Each checkpoint stores its own $\mu$, $C$ and feature statistics, so the average is taken in
$\psi$-space. The ported inference reproduces the original ensemble output with maximum absolute difference 0 on the three
demo shots.
