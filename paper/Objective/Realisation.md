# How the model components realise the objective

The two components address different parts of the objective.

* **M9** trained with coefficient MSE over 60 principal components of EFIT++ flux (Section 4)
  minimises an intensity term; the network sees no boundary, point or scalar term, and its training
  records only the validation coefficient MSE.
* **OSF** imposes Grad–Shafranov physics, which fixes the relation between currents, field and
  boundary, and reads $q_{95}$ and $\beta_N$ off the solved equilibrium (Section 3). Its calibrated
  parameters were set by hand from physics end-members on the three validation shots, inspected in
  the viewer (Section 5.3); this is the route applied by eye.
* **Selection** by coefficient MSE or by a flux threshold on a few shots uses the target, not the
  route.

The audit of Section 7.2 is the formal version of the viewer inspection (Section 7.5); putting
$\mathcal R$ into training and checkpoint selection is the regime of Section 10.6.
