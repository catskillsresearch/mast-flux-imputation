# Magnetic topology: O-point, X-points, LCFS

The magnetic axis is the maximum of $\psi$ (since $\sigma_{\rm ax}=+1$) inside the box
$R\in[0.65,1.05]$ m, $|Z|\le0.5$ m. During the iteration the last closed surface is the outermost
compact closed contour around the axis (elongation $\le2.6$, vertical span $\le2.8$ m, outer radius
$\ge1.0$ m, inner radius $\le0.4$ m); candidate X-points are saddles in $R\in[0.25,1.5]$ m with
$|Z|\ge0.4$ m.

For the read-outs, OSF calls the *scorer's* LCFS extractor (`fec_scoring/lcfs.py`, marching squares
via scikit-image [@vanderwalt2014]) restricted by the MAST vessel envelope
`mast_envelope.npz` (`mask_coarse`). The envelope mask is
used here in the $q_{95}$ and $\beta_N$ read-outs only (a clarification of the earlier documentation). If the scorer extractor fails,
OSF falls back to its own closed-contour search.
