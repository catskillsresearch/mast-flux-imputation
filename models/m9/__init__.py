"""Component 2 -- M9 (``M9_win50``): archive-trained flux predictor.

A 60-mode PCA subspace for psi(R,Z), fitted on FAIR-MAST EFIT++ ``efit_psirz`` frames, plus a
dual-context residual MLP that maps [coil currents, I_p, core Thomson T_e/n_e channels] with a
9-frame temporal context to the 60 PCA coefficients. Three members (torch init seeds 101, 202,
303; identical data split seed 42) are averaged in psi space. See docs/MODEL.md §M9.
"""
from models.m9.predict import M9Ensemble, MEMBER_TAGS

__all__ = ["M9Ensemble", "MEMBER_TAGS"]
