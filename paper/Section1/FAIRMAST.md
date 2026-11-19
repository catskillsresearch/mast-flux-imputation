# FAIR-MAST archive and ingest

FAIR-MAST publishes the MAST archive as Zarr stores at `s3://mast/level1/shots/N.zarr` for shot
number N (endpoint `s3.echo.stfc.ac.uk`, anonymous, CC BY-SA 4.0). The ingest
(`data/fair_mast/ingest.py`) reads four groups per shot,

| group | content | use |
|---|---|---|
| `efm` | EFIT++: `psirz`, grid `r`, `z`, `q_95`, `betan`, X-points | labels only |
| `amc` | coil and plasma currents | inputs |
| `ayc` | core Thomson $T_e, n_e$ | inputs |
| `aye` | edge Thomson $T_e, n_e$ | inputs (zeroed by the model) |

and writes one row per shot in the template layout, with `efit_psirz` and the EFIT++ grid. 3,738
shots (numbers 23240–30471) were ingested, keeping all EFIT slices (ramp-up, limited, ramp-down).

**No-plasma shots.** 349 of the 3,738 shots carry a reduced EFIT++ product: 200 slices on a fixed
5 ms clock (0–995 ms or $-50$–945 ms), 60 instead of about 150 `efm` variables, no scalar or
X-point outputs, and $\max|I_p|\le46.8$ kA (median 9.1 kA). They are shots without a plasma,
spread over the whole shot range; 335 lie in the training pool and 14 in the sealed holdout.

`data/fair_mast/fetch_efit_labels.py` adds the scalar labels `q_95`, `betan` and the diverted flag
(any finite `efm/xpoint{1,2}_rc`) for each EFIT slice, again from FAIR-MAST.
