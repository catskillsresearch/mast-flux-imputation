# Hypothesis

Let a model be trained by minimising a simple loss $\mathcal L$ that is close to one target term,
here coefficient MSE, which is close to $1-R^2_\psi$ restricted to the PCA subspace. The hypothesis
is:

> **H.** Minimising $\mathcal L$ can give a high target score $S$ on the data used for fitting or
> selection while the route quantities $\overline m_k$ stay large; such a model generalises less
> well, in $S$ and in the route, than one whose training or selection also reduces $\mathcal R$.

The mechanism is that $R^2_\psi$ is pooled over pixels and dominated by the vacuum field, so a
residual concentrated near the boundary, where the LCFS, X-points, $\delta$ and $q_{95}$ are
decided, costs little in $\mathcal L$ but a lot in geometry and in the scalar terms of $S$. Matching
the observables constrains exactly that residual.
