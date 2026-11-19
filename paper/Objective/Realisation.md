# How the current models realise the objective

The two models address different parts of the objective, and neither is trained on $\mathcal R$.

* **M9** minimises coefficient MSE over 60 principal components of EFIT++ flux (Section 4). That
  is an intensity term; the network sees no boundary, point or scalar term. Its training records
  only the validation coefficient MSE, so no route quantities exist for its training data.
* **OSF** imposes Grad–Shafranov physics, which fixes the relation between currents, field and
  boundary, and reads $q_{95}$ and $\beta_N$ off the solved equilibrium (Section 3). Its calibrated
  parameters were set by hand from physics end-members on the three validation shots, inspected in
  the viewer, and from five aggregate-score probes (Section 5.3); this is the route applied by eye.
* **Selection** of the shipped M9 checkpoints used a target threshold on the three validation shots
  ($R^2_\psi\ge0.9976$), not route quantities.

The audit of Section 7.2 is the formal version of the viewer inspection. It now scores both models
(Section 7.5); putting $\mathcal R$ into training and checkpoint selection is the regime of
Section 10.6.
