# Picard iteration

Because $p'$ in \eqref{eq:jphi} depends on $\Delta\psi$ and $s_V$, and the read-outs depend on the
flux-surface geometry, the solve is a fixed-point (Picard) iteration. Starting from
$\psi^{(0)}=\psi_{\rm vac}$, pass $n$:

1. locate the O-point and the last closed surface of $\psi^{(n-1)}$; if they are acceptable,
   set $\Delta\psi$ from them and $W=\oint d\ell/(R|\nabla\psi|)$ on the 95% surface, otherwise
   fall back to the maximum of $\psi$ inside $\Omega_p$ and the mean outside;
2. evaluate $j_\varphi$ from \eqref{eq:jphi} and rescale it with \eqref{eq:ampere};
3. update $\psi^{(n)} = \psi_{\rm vac} + G\,j_\varphi$.

The loop stops when $\max|\psi^{(n)}-\psi^{(n-1)}|<10^{-4}$ Wb rad$^{-1}$ *and* the relative
changes of $W$, $\Delta\psi$ and $s_V$ are all below $10^{-3}$, or after 20 passes. Earlier project
documents called OSF "one-shot"; that is accurate only in the sense that nothing is fitted to
measured flux — numerically it is an iteration of up to 20 linear solves per frame.
