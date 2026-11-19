# Sources

Every shot used to train, select or evaluate the model is downloaded from FAIR-MAST
[@jackson2024fairmast], the open archive of the MAST spherical tokamak published by UKAEA. The
ground truth for every frame is the EFIT++ reconstruction [@appel2006] stored in the archive with
the shot.

The FEC MAST task [@fec2026dataset; @fec2026starter] (CC BY 4.0) supplies the scoring code and the
output template. The template holds one Parquet row per shot with nested arrays for every time
series; the prediction is $\psi$ on a fixed $65\times65$ $(R,Z)$ grid, $q_{95}$ and $\beta_N$ at each
scored EFIT time. Only *diverted* EFIT time slices are scored; a MAST slice is diverted when
EFIT++ reports an X-point (`efm/xpoint{1,2}`). The input time series are not subset, so the scored
`efit_times` need not be contiguous.
