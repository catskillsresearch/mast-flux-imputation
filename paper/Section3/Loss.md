# Model and loss

M9 is a PCA basis of EFIT++ flux combined with an ensemble of three residual multilayer
perceptrons that map the measured inputs to the 60 PCA coefficients. The base recipe
trains them with the mean-squared error between predicted and EFIT++ coefficients,

$$
\mathcal L(\theta) = \frac{1}{|B|}\sum_{t\in B}\frac{1}{60}\big\|f_\theta(\tilde x_t) - z_t\big\|_2^{2},
\qquad z_t = C\,(\psi^{\rm EFIT}_t-\mu).
$$

The physical structure of the output comes from the EFIT++ training targets and from the PCA
subspace they span. The network also has five scalar heads ($q_{95}$, $\beta_N$, $R_{\rm axis}$,
$Z_{\rm axis}$, $\ell_i$). With the base recipe they carry no loss; from experiment E3 on (Section 10.6)
they are trained on EFIT++ labels and supply $q_{95}$ and $\beta_N$.
