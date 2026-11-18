# Current-density ansatz and Ampère constraint

**Support.** The plasma current is confined to a fixed ellipse $\Omega_p$ (the alternative D-shaped
support of the source code is disabled for MAST, `USE_VACUUM_XD = False`):

$$
\begin{aligned}
R_0 &= 0.82+0.10\,w-0.02\,b, & Z_0 &= 0.009-0.004\,b,\\
a_R &= 0.50+0.08\,w, & a_Z &= a_R+0.08+0.42\,b+0.18\,(1-b),
\end{aligned}
$$

all in metres, with the geometric normalised radius
$\tau=\rho^2=((R-R_0)/a_R)^2+((Z-Z_0)/a_Z)^2\in[0,1]$ on $\Omega_p$. $\tau$ is geometric, not a
function of $\psi$: the current shape is held fixed so that it cannot chase an outboard maximum.

**Profiles.** With $s_V$ the volume average over $\Omega_p$ of $s(\tau)=\tfrac12(1-\tau)^2$
(the integral of the pressure-derivative shape from $\tau$ to 1) and
$\Delta\psi=\psi_{\rm ax}-\psi_b$,

$$
p'(\tau) = \frac{\langle p\rangle\,(1-\tau)}{\Delta\psi\,s_V},\qquad
FF'(\tau) = -\bar\gamma\,(1-\tau),\quad \bar\gamma=-1,\qquad
j_\varphi = R\,p' + \frac{FF'}{\mu_0R}\ \text{ on }\Omega_p .
\label{eq:jphi}
$$

The pressure scale comes from a first-guess normalised beta $\beta_N^{(0)}$,

$$
\langle p\rangle = \frac{\beta_N^{(0)}}{100}\,\frac{B_t}{2\mu_0}\,\frac{I_p[\mathrm{MA}]}{a_R},
$$

where $\beta_N^{(0)}$ is
computed from the median Thomson pressure $2n_eT_e e$ over chords with $R\in[0.25,1.45]$ m, clipped to
$[0.2,4]$ (default 1.3 when no chord is valid).

**Ampère constraint.** The amplitude is then rescaled so that the enclosed current matches a
calibrated fraction of the measured plasma current:

$$
\int_{\Omega_p} j_\varphi\,dA = k\,I_p,\qquad k = 0.70+0.30\,w+0.15\,b+0.20\,(1-b).
\label{eq:ampere}
$$

The factor $k\ne1$ absorbs, in an averaged way, the current that a fixed ellipse misplaces and the
missing vessel currents. Only the *ratio* of the $p'$ and $FF'$ terms survives the rescaling, so the
pressure guess shapes the current but does not set its total.
