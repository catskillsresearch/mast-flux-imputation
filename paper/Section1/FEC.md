# Fusion Equilibrium Challenge MAST data

The FEC dataset [@fec2026dataset] (CC BY 4.0) stores one Parquet row per shot with nested arrays
for every time series. Its MAST portion used here consists of

* `mast_public_test`: 1,206 shots with inputs only (labels withheld for the leaderboard);
* three labelled starter-kit shots, 28348, 28350 and 28351 (36, 42 and 37 EFIT frames), distributed
  with the starter kit [@fec2026starter] and fetched with `data/fec/fetch.py --demo` (sha256-pinned).

FEC keeps only *diverted* EFIT time slices. For MAST a slice is diverted when EFIT++ reports an
X-point (`efm/xpoint{1,2}`); the challenge reports that this keeps 66.4% of MAST frames
[@fec2026dataset]. The input time series are not subset, so `efit_times` is not contiguous.
