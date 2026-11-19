# Weighting

Two weights concentrate the curve distance where the errors matter most.

* **Late weight.** With $t_{\rm frac}\in[0,1]$ the position of a frame within its shot,
  $$
  w_{k,t} = 1 + \alpha_k\,\mathrm{clip}\Big(\frac{t_{\rm frac}-0.8}{0.2},\,0,\,1\Big),
  $$
  rising linearly over the last 20% of the shot, with $\alpha_k=4$ for $\beta_N$ and 2 for the other
  series. The end of the shot, where $\beta_N$ falls quickly, is otherwise a few frames among many.
* **Series weight.** $\omega_k=2$ for $\beta_N$, $\delta_{\rm top}$ and $\delta_{\rm bot}$ and 1 for
  the other five series.

The validation objective uses the same weights and adds the soft-LCFS term with $\alpha=2$ and
$\omega_{\rm LCFS}=1$.
