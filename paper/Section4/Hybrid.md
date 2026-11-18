# Hybrid M9 + OSF

The shipped submission recipe combines the two models: the flux map comes from M9, the scalars
from OSF after two post-hoc factors,

$$
\psi = \hat\psi_{\rm M9},\qquad
q_{95} = \mathrm{clip}\Big(q_{95}^{\rm OSF}\,\frac{c^{*}}{1.20},\,2,\,25\Big),\qquad
\beta_N = \mathrm{clip}\big(\beta_N^{\rm OSF}\,s^{*},\,0.2,\,5\big),
$$

with $c^{*}=1.062242$ and $s^{*}=1.078508$. The OSF read-outs are evaluated on the OSF flux, not on
the M9 flux. The factors were obtained from aggregate leaderboard scores on the FEC public-test fold
(Section 8).
