# Loss

For a batch of whole shots with frames $t$, predicted coefficients $\hat z_t$, EFIT++ coefficients
$z_t$ and heads $\hat u_t$, the loss is

$$
\begin{aligned}
\mathcal L ={}& \overline{\|\hat z_t-z_t\|^2}
 + w_{cd}\,\mathcal L_{cd} + w_{\rm LCFS}\,\overline{\big(1-\mathrm{IoU}_{\rm soft}\big)}
 + w_b\,\mathcal L_{b} + w_{\rm tip}\,\mathcal L_{\rm tip}\\
&+ w_c\,\mathcal L_{c} + w_t\,\overline{\|(\hat z_t-\hat z_{t-1})-(z_t-z_{t-1})\|^2}
 + w_s\sum_j \mathrm{Huber}\big(\hat u_{j,t}-u_{j,t}\big).
\end{aligned}
\label{eq:loss}
$$

* **Coefficient anchor.** MSE on the 60 coefficients, i.e. pixel MSE of the projected target.
* **Curve distance** $\mathcal L_{cd}$, the main calibration term. For each series
  $k\in\{q_{95},\beta_N,\kappa,\delta_{\rm top},\delta_{\rm bot},V,R_{\rm axis},Z_{\rm axis}\}$ the error
  $e_{k,t}=(\hat u_{k,t}-u_{k,t})/r_{k,n}$ is normalised by the range $r_{k,n}$ of the EFIT++ trace
  over shot $n$, floored at 0.25 of the series' global standard deviation. $q_{95}$ and $\beta_N$ come
  from the heads, the six shape series from the soft read-outs of $\hat\psi$. Per shot,
  $$
  d_{k,n} = \frac{\sum_t w_{k,t}\,e_{k,t}^2}{\sum_t w_{k,t}}
  + \lambda_d\,\frac{\sum_t w_{k,t}\,(e_{k,t}-e_{k,t-1})^2}{\sum_t w_{k,t}},\qquad \lambda_d=0.5,
  $$
  with the late weights $w_{k,t}$ of the next note, and
  $\mathcal L_{cd}=\sum_k\omega_k\,\overline{d_{k,n}}\big/\sum_k\omega_k$, the bar averaging over shots.
* **Soft LCFS.** One minus the soft intersection-over-union of the soft interior $s$ and the EFIT++
  LCFS interior, $w_{\rm LCFS}=w_{cd}/3$.
* **Boundary band** $\mathcal L_b$: pixel error inside $|\psi_N-1|<0.1$ of the EFIT++ LCFS;
  **tip band** $\mathcal L_{\rm tip}$: the same restricted to the top and bottom 15% of the EFIT++
  LCFS height, where the scorer reads $\delta_{\rm top}$ and $\delta_{\rm bot}$.
* **Critical points** $\mathcal L_c$: $|\nabla\hat\psi|^2$ at the EFIT++ O- and X-points and the
  X-point flux mismatch.
* **Temporal term** on frame-to-frame coefficient differences.
* **Scalar heads:** Huber loss ($\delta=1$) on the five standardised EFIT++ scalars.

| weight | $w_{cd}$ | $w_{\rm LCFS}$ | $w_b$ | $w_{\rm tip}$ | $w_c$ | $w_t$ | $w_s$ |
|---|---|---|---|---|---|---|---|
| value | 0.01 | 0.00333 | 3 | 10 | $10^{-3}$ | 1 | $3\times10^{-3}$ |

Table: Loss weights.
