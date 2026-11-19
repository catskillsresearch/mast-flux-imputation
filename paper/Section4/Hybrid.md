# Hybrid M9 + OSF

The model combines the two components. The flux map is the M9 ensemble mean, $\psi=\hat\psi_{\rm M9}$.
In the MSE-only form of the model (experiments E1 and E2 of Section 10.6) $q_{95}$ and $\beta_N$
come from the OSF read-outs through affine maps fitted to FAIR-MAST EFIT++ labels on 200 training
shots (experiment E5),

$$
q_{95} = \mathrm{clip}\big(a_q\,q_{95}^{\rm OSF}+b_q,\,2,\,25\big),\qquad
\beta_N = \mathrm{clip}\big(a_b\,\beta_N^{\rm OSF}+b_b,\,0.2,\,5\big),
$$

with $a_q=0.626$, $b_q=1.957$, $a_b=1.013$, $b_b=0.093$. The OSF read-outs are evaluated on the OSF
flux, not on the M9 flux. From experiment E3 on, the network's scalar heads, trained on EFIT++
labels, supply $q_{95}$ and $\beta_N$; in that form OSF is not evaluated at inference and serves as
the physics reference, the source of the read-out baseline and the comparison model of the audit.
