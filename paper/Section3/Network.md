# Network and scalar heads

Each member $f_s$ (`CoeffDualContext`) has two residual encoders [@he2016]:

$$
\begin{aligned}
h &= \mathrm{Res}^{(4)}_{512}\big(\mathrm{ReLU}(W_h\,\tilde x_{[0,308)})\big), &
c &= \mathrm{Res}^{(3)}_{512}\big(\mathrm{ReLU}(W_c\,\tilde x_{[308,924)})\big),\\
h' &= h + g\,P\,c, & z &= W_z\,h' + b_z\in\mathbb R^{60},
\end{aligned}
\label{eq:net}
$$

where $\mathrm{Res}^{(n)}_{512}$ is a stack of $n$ residual blocks of width 512 with dropout 0.05,
$P$ is a $512\times512$ linear projection and $g$ is a scalar gate initialised to 0, so training
starts from the present-frame encoder alone and learns how much temporal context to admit. The
predicted flux is reconstructed with \eqref{eq:pca}.

Five linear heads read the same representation,

$$
\hat u_j = w_j^{\top}h' + b_j,\qquad j\in\{q_{95},\ \beta_N,\ R_{\rm axis},\ Z_{\rm axis},\ \ell_i\},
$$

each trained on the EFIT++ value standardised with its training mean and standard deviation. The
$q_{95}$ and $\beta_N$ heads, de-standardised, are the model's $q_{95}$ and $\beta_N$ outputs. The
axis and $\ell_i$ heads are trained as auxiliary targets; in scoring, these quantities are derived
from the predicted flux map.
