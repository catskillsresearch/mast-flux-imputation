# Abstract

We reconstruct the poloidal magnetic flux $\psi(R,Z)$ of MAST plasmas on a $65\times65$ grid, together
with the edge safety factor $q_{95}$ and the normalised beta $\beta_N$, using only the poloidal-field
coil currents, the plasma current $I_p$, the toroidal-field current and midplane Thomson scattering.
No magnetic probe or flux-loop signal enters either model. Two models are described. OSF is a
free-boundary Grad–Shafranov forward solver: an elliptic-integral Green's-function vacuum field over
the 812-element MAST coil set, a prescribed-shape current density fixed by an Ampère constraint and
iterated to self-consistency, a flux-surface integral for $q_{95}$ and a Thomson pressure moment for
$\beta_N$. M9 is a PCA-basis neural surrogate: a residual network maps measured inputs to 60
principal-component coefficients of EFIT++ flux, trained on 3,622 FAIR-MAST archive shots; despite
its legacy file name it is not physics-informed. On a sealed, seed-42 holdout of FAIR-MAST shots
disjoint from M9 training (65 shots, 3,449 diverted frames), scored with the Fusion Equilibrium
Challenge metric, OSF reaches $S=0.618634$ ($R^2_\psi=0.917210$) and the hybrid M9 + OSF reaches
$S=0.862677$ ($R^2_\psi=0.997771$, $1-D_{\rm LCFS}=0.945207$), while $q_{95}$ and $\beta_N$ remain weak
($R^2=0.342298$ and $0.394260$). We document exactly what data each component was trained or tuned on, and propose an automated failure-mode
audit to replace visual inspection when training at archive scale.
