# LCFS distance

For each frame the last closed flux surface $\Gamma$ is extracted from the predicted and from the
EFIT++ map by the same code (marching squares inside the vessel envelope). The per-frame distance is
the symmetric Hausdorff distance [@huttenlocher1993] normalised by the reference geometric major
radius of the shot,

$$
d_t = \min\Big(1,\ \frac{d_H(\Gamma^{\rm pred}_t,\Gamma^{\rm EFIT}_t)}{R_{\rm geo}}\Big),\qquad
d_H(A,B)=\max\Big\{\sup_{a\in A}\inf_{b\in B}|a-b|,\ \sup_{b\in B}\inf_{a\in A}|a-b|\Big\}.
$$

$D_{\rm LCFS}$ averages $d_t$ over the frames of a shot and then over shots. A failed predicted
LCFS counts as $d_t=1$; frames whose *reference* LCFS cannot be extracted are dropped.
