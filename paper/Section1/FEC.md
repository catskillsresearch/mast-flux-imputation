# Output format and validation shots

The output format follows the FEC MAST task [@fec2026dataset] (CC BY 4.0): one Parquet row per
shot with nested arrays for every time series, the EFIT++ flux on a fixed $65\times65$ $(R,Z)$ grid,
and $q_{95}$ and $\beta_N$ at each EFIT time. Only *diverted* EFIT time slices are scored; for MAST a
slice is diverted when EFIT++ reports an X-point (`efm/xpoint{1,2}`). The input time series are
not subset, so `efit_times` is not contiguous.

The three labelled shots of the FEC starter kit [@fec2026starter], 28348, 28350 and 28351 (36, 42
and 37 diverted frames), are the validation shots. They are fetched with
`data/fec/fetch.py --demo` (sha256-pinned) and are excluded from the FAIR-MAST training pool, the
quality-control statistics and the PCA basis.
