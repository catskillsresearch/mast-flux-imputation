# Problem statement

Equilibrium reconstruction recovers the axisymmetric magnetic equilibrium of a tokamak plasma from
measurements. The standard tool, EFIT [@lao1985] and its MAST implementation EFIT++ [@appel2006],
fits the Grad–Shafranov equation [@shafranov1958; @grad1958] to magnetic probes and flux loops,
optionally constrained by kinetic and internal-field data. The magnetic sensors carry most of the
information about the plasma boundary.

The Fusion Equilibrium Challenge (FEC) [@fec2026dataset; @fec2026starter] poses the inverse
question: how much of $\psi(R,Z)$ can be recovered *without* direct magnetic sensors? Its MAST
[@darke1995; @sykes2001; @lloyd2003] inputs are the actuator currents (solenoid, P2–P6 upper/lower
poloidal-field circuits, the toroidal-field coil and the EFPS circuit), the plasma current $I_p$,
and the core and edge Thomson scattering profiles $T_e(R), n_e(R)$ on the horizontal midplane
[@scannell2010]. The targets are the EFIT++ flux map on a fixed $65\times65$ $(R,Z)$ grid,
$q_{95}$ and $\beta_N$ at each EFIT time.

This paper describes, at the level of the code, a MAST-only system for this task extracted into
the repository `catskillsresearch/mast-flux-imputation`, and reports its measured performance on a
sealed archive holdout.
