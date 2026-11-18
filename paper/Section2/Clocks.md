# Walk and bleed clocks

A MAST discharge moves from a solenoid-driven start-up towards a diverted flat-top. OSF summarises
the phase with two dimensionless "clocks" computed from the measured currents at each EFIT time:

$$
w = \Big(1-\min\Big(\frac{\bar I_{P2}}{|I_p|\,r_\star},\,1\Big)\Big)^{2},\qquad
\bar I_{P2}=\tfrac12\big(|I_{P2U}|+|I_{P2L}|\big),\qquad
b = \mathrm{clip}\Big(\frac{I_{\rm sol}}{I_\star},0,1\Big),
\label{eq:clocks}
$$

with $r_\star=0.34$ and $I_\star=16.3$ kA. The divertor "walk" $w$ approaches 1 as the P2 current
becomes small relative to $I_p$; the solenoid "bleed" $b$ is 1 while the solenoid still carries
positive current and 0 once it has swung through zero. In the code these clocks set the current
support, the coil multipliers, the enclosed-current factor and the read-out laws. They are not
fitted per shot; they are deterministic functions of the actuators, which is how OSF blends
limited-like and diverted-like behaviour without a classifier.
