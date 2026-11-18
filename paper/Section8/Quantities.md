# Per-frame audit quantities

For every frame of every FAIR-MAST shot, compute from the model and EFIT++ maps, with the scorer's
own extractors:

* LCFS symmetric Hausdorff distance and the area of the symmetric difference of the two enclosed
  regions, normalised by the EFIT++ area;
* O-point displacement $|\Delta(R,Z)_{\rm axis}|$ and X-point displacement for each active X-point;
* topology flags: X-point lost or gained, and limited/diverted flips relative to EFIT++;
* errors of $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$, $V$ and $\ell_i$;
* the fraction of the residual outside the 60-mode PCA subspace (what M9 cannot represent);
* for OSF, the Ampère ratio and the number of Picard passes.
