# Differentiable shape read-outs

The scorer derives elongation, triangularity, volume and the magnetic axis from contours of the
flux map, which are not differentiable. For training, the same quantities are read from the
predicted $\hat\psi$ by soft extractors (`experiments/e6.py`, `soft_feats`):

* the flux is normalised with the EFIT++ axis and boundary values,
  $\psi_N=(\psi_{\rm ax}-\hat\psi)/(\psi_{\rm ax}-\psi_b)$, and the plasma interior is the soft level
  set $s=\sigma\big((1-\psi_N)/\tau\big)$, $\tau=0.02$, restricted to the EFIT++ LCFS interior dilated
  by two cells (the vessel-envelope mask of the scoring code where no EFIT++ LCFS is available);
* each grid column gives a sub-grid top and bottom of the interior (centroid $\pm$ half the soft
  height) and each grid row a sub-grid inner and outer edge; $Z_{\max}$, $Z_{\min}$, $R_{\max}$,
  $R_{\min}$ and the $R$ of the highest and lowest points are soft-argmax read-outs over those
  columns and rows (temperature 0.03 m);
* $\kappa=(Z_{\max}-Z_{\min})/(R_{\max}-R_{\min})$, $\delta_{\rm top}=(R_{\rm geo}-R_{Z_{\max}})/a$,
  $\delta_{\rm bot}=(R_{\rm geo}-R_{Z_{\min}})/a$ with $a=(R_{\max}-R_{\min})/2$ and
  $R_{\rm geo}=(R_{\max}+R_{\min})/2$;
* $V=2\pi\sum R\,s\,\Delta A$, and the axis is the soft-argmin of $\psi_N$ (temperature 0.02).

The read-outs are checked against the scorer's extractor in two ways: on EFIT++ maps, by the
correlation of the soft and the scorer values; on predicted maps, by the correlation of the soft
error and the scorer error against EFIT++.

@import "../generated/model_softcheck.md"

Table: Soft read-outs against the scorer's extractor on archive validation frames.

On EFIT++ maps all six read-outs track the scorer closely. The errors track it well for $V$ and
the axis, moderately for $\kappa$, and hardly for $\delta_{\rm top}$ and $\delta_{\rm bot}$, whose
scorer values depend on where the contour extractor places the tip of the LCFS.
