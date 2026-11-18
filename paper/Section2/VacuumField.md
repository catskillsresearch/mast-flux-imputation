# Vacuum flux from the coil set

The flux per radian at $(R,Z)$ from a coaxial circular filament of radius $R'$ at height $Z'$
carrying current $I$ is [@jacksonCED1999; @lackner1976]

$$
\psi = I\,G(R,Z;R',Z'),\qquad
G = \frac{\mu_0\sqrt{RR'}}{2\pi k}\Big[(2-k^{2})K(k)-2E(k)\Big],\qquad
k^{2}=\frac{4RR'}{(R+R')^{2}+(Z-Z')^{2}},
\label{eq:green}
$$

with $K$ and $E$ the complete elliptic integrals of the first and second kind
[@abramowitz1964] (SciPy `ellipk`, `ellipe` with parameter $m=k^2$). $G$ is set to 0 on the axis
$R=0$ and $k^2$ is clipped below 1 to avoid the logarithmic singularity at coincident points.

The FEC MAST coil table has 812 rectangular elements: 656 solenoid rows, each one turn carrying
$I_{\rm sol}$, and 156 elements of the P2–P6 upper/lower circuits, each carrying its circuit's
ampere-turns divided by the number of elements of that circuit. Each rectangle of width $w$ and
height $h$ is replaced by the mean of $3\times3$ filaments at offsets $\{-1,0,1\}\cdot w/3$,
$\{-1,0,1\}\cdot h/3$. This gives one unit map $\Psi_c(R,Z)$ per circuit $c$, and

$$
\psi_{\rm vac}(R,Z) = \sum_{c}\lambda_c(b)\,I_c\,\Psi_c(R,Z).
\label{eq:vac}
$$

The TF and EFPS circuits do not enter $\psi_{\rm vac}$. The multipliers $\lambda_c$ are calibrated
functions of the bleed clock $b$ (next note):

| circuit | multiplier |
|---|---|
| P2 upper and lower | $\lambda_V = 1.8 - b$ |
| P2 lower, additionally | $\times\,(1+0.105(1-b))$ |
| P4 pair | $1.15$ |
| P5 pair | $\times\,(1-0.18(1-b))$ |
| solenoid | $0.5+0.10\,b$ |
| P3, P6 | 1 |

The plasma term in \eqref{eq:freeb} uses the same kernel evaluated between grid nodes, assembled once
as a $4225\times4225$ matrix including the cell area $\Delta R\,\Delta Z$.
