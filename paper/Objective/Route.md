# Route: observable distances

The route consists of the quantities a physicist reads off the viewer `app/fec_site` for each
frame $t$ of a shot: the flux map with its last closed flux surface (LCFS) $\Gamma_t$, enclosed region
$\Omega_t$, O-point $x_{O,t}$ and X-points $x_{X,t}$, and the scalar traces below it. With hats for
the prediction and no hat for EFIT++, the audit `pipeline/audit.py` computes, per frame, the intensity terms

$$
e_{\psi,t}=\frac{1}{N}\sum_{ij}\big(\hat\psi_{t,ij}-\psi_{t,ij}\big)^2,\qquad
e_{s,t}=\big|\hat s_t-s_t\big|,\quad s\in\{q_{95},\beta_N,R_{\rm axis},Z_{\rm axis},\kappa,
\delta_{\rm top},\delta_{\rm bot},V,\ell_i\},
\label{eq:route-intensity}
$$

the geometry terms

$$
h_t=d_H(\hat\Gamma_t,\Gamma_t),\qquad a_t=\frac{|\hat\Omega_t\,\triangle\,\Omega_t|}{|\Omega_t|},\qquad
o_t=|\hat x_{O,t}-x_{O,t}|,\qquad x_t=|\hat x_{X,t}-x_{X,t}|,
\label{eq:route-geometry}
$$

and the indicators $j_t=[\,g_t\ge 50\ \mathrm{mm}\,]$ with
$g_t=\max(|\Delta Z_{\rm top}|,|\Delta Z_{\rm bot}|)$ the vertical gap of the LCFS tips (the
viewer's jump flag), X-point lost or gained, and limited/diverted flip. Per shot, each scalar trace
also gets a lag $\ell_s=\arg\max_{|L|\le4}\mathrm{corr}(\hat s_{t+L},s_t)$, a jitter ratio
$J_s=\mathrm{rms}(\Delta^2\hat s)/\mathrm{rms}(\Delta^2 s)$ and a linear drift of $\hat s-s$.

The route functional is a weighted sum of fold means of these quantities, each scaled by a
reference value $m_k^{\rm ref}$ (for example the value of the current model),

$$
\mathcal R=\sum_k w_k\,\frac{\overline{m}_k}{m_k^{\rm ref}} .
\label{eq:route}
$$

The weights $w_k$ are a design choice of the planned training regime and are not fixed here. For
comparing two models the audit avoids them: it reports every $\overline m_k$ and every target term
separately, and a new model is accepted only if it is at least as good as the old one on each.
