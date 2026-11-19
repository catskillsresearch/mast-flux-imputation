# Problem statement

Equilibrium reconstruction recovers the axisymmetric magnetic equilibrium of a tokamak plasma from
measurements. The standard tool, EFIT [@lao1985] and its MAST implementation EFIT++ [@appel2006],
fits the axisymmetric force-balance equation to magnetic probes and flux loops, optionally
constrained by kinetic and internal-field data. The magnetic sensors carry most of the information
about the plasma boundary.

This paper asks how much of $\psi(R,Z)$ can be recovered *without* direct magnetic sensors. The
inputs are the MAST [@darke1995; @sykes2001; @lloyd2003] actuator currents (solenoid, P2–P6
upper/lower poloidal-field circuits, the toroidal-field coil and the EFPS circuit), the plasma
current $I_p$, and the core Thomson-scattering profiles $T_e(R), n_e(R)$ on the horizontal midplane
[@scannell2010]. The targets are the EFIT++ flux map on a fixed $65\times65$ $(R,Z)$ grid, $q_{95}$
and $\beta_N$ at each EFIT time.

All data used to train, select and evaluate the model come from the open FAIR-MAST archive
[@jackson2024fairmast]. The output template and the scoring metric are those of the FEC MAST task
[@fec2026dataset; @fec2026starter]. The model, its calibration and the evaluation are
in the repository `catskillsresearch/mast-flux-imputation`.
