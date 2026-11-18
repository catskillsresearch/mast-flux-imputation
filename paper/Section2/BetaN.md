# Normalised beta read-out

The Troyon-normalised beta [@troyon1984] is
$\beta_N = 100\,\beta_t\,a\,B_t/I_p[\mathrm{MA}]$ with $\beta_t = 2\mu_0\langle p\rangle/B_t^{2}$, i.e.

$$
\beta_N = \frac{100\,\langle p\rangle\,2\mu_0\,a}{B_t\,I_p[\mathrm{MA}]},\qquad B_t = F_b/R_0 .
\label{eq:betan}
$$

OSF estimates $\langle p\rangle$ as the mean of $2n_eT_e e$ (equal ion and electron pressure assumed)
over Thomson chords, interpolated to the EFIT time, whose radius lies inside the extracted LCFS,
$R\in[r_{\rm in},r_{\rm out}]$, with $a=(r_{\rm out}-r_{\rm in})/2$. Per frame the value is
clipped to $[0.2,4]$; over the shot a 7-frame running median removes single laser spikes, and the
result is multiplied by $(1-0.2\,w)$. The pressure is a midplane line average, not a volume average,
which is one reason the $\beta_N$ skill is limited (Section 7).
