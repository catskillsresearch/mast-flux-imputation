# What M9 is, and is not

The training script of the source repository is named `run_crc_feq_pinn_mast.py`, and the model was
informally called a "PCA-PINN". It is not a physics-informed neural network in the sense of
[@raissi2019]: the loss is the mean-squared error between predicted and EFIT++ PCA coefficients,

$$
\mathcal L(\theta) = \frac{1}{|B|}\sum_{t\in B}\frac{1}{60}\big\|f_\theta(\tilde x_t) - z_t\big\|_2^{2},
\qquad z_t = C\,(\psi^{\rm EFIT}_t-\mu),
$$

with no Grad–Shafranov residual, boundary condition or conservation term. Any physics in the output
is inherited from the EFIT++ training targets and from the PCA subspace they span. We therefore call
M9 a *PCA-basis neural surrogate*; the "pinn" in the file name is legacy. The network also carries
five scalar heads ($q_{95}$, $\beta_N$, $R_{\rm axis}$, $Z_{\rm axis}$, $\ell_i$) that receive no loss
and are kept only so that the shipped state dictionaries load.
