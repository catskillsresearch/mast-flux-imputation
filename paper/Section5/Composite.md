# Composite score

For a fold of shots, with $R^2 = 1-\mathrm{SS}_{\rm res}/\mathrm{SS}_{\rm tot}$ pooled over all
frames (and, for $\psi$, all pixels) of the fold and $\mathrm{SS}_{\rm tot}$ taken about the
ground-truth mean,

$$
S = 0.55\,\max(0,R^2_\psi) + 0.15\,\max\Big(0,\tfrac12\big(R^2_{q_{95}}+R^2_{\beta_N}\big)\Big)
+ 0.10\,\big(1-\min(1,D_{\rm LCFS})\big) + 0.20\,\mathrm{Cons}.
\label{eq:score}
$$

Pooling over the fold means that shots with many frames weigh more, and that $R^2$ is measured
against the variance *across* shots, which is large for $\psi$ and comparatively small for
$q_{95}$ and $\beta_N$.
