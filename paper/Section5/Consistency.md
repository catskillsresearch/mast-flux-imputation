# Consistency scalars

Seven scalars are derived from each predicted and each EFIT++ map by the same routine
(`fec_scoring/derive.py`): the magnetic-axis position $R_{\rm axis}, Z_{\rm axis}$; elongation
$\kappa$ and upper/lower triangularity $\delta_{\rm top},\delta_{\rm bot}$ from the LCFS extremes;
the plasma volume $V = 2\pi\int_A R\,dR\,dZ$ inside the LCFS; and the internal inductance in the
ITER form $\ell_i(2)=\langle B_p^{2}\rangle_V/(\mu_0I_p/\ell_p)^{2}$, evaluated from $|\nabla\psi|$.
With $R^2_j$ the pooled $R^2$ of scalar $j$,

$$
\mathrm{Cons} = \frac17\sum_{j=1}^{7}\max\big(0,R^2_j\big).
$$

The term measures whether the predicted map is geometrically equivalent to the reconstruction, not
whether it is close pixel by pixel.
