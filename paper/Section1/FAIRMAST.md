# FAIR-MAST archive and ingest

FAIR-MAST [@jackson2024fairmast] publishes the MAST archive as Zarr stores at
`s3://mast/level1/shots/N.zarr` for shot number N (endpoint `s3.echo.stfc.ac.uk`, anonymous, CC BY-SA 4.0). The
ingest (`data/fair_mast/ingest.py`) reads four groups per shot,

| group | content | use |
|---|---|---|
| `efm` | EFIT++: `psirz`, `q_95`, `betan`, X-points | labels only |
| `amc` | coil and plasma currents | inputs |
| `ayc` | core Thomson $T_e, n_e$ | inputs |
| `aye` | edge Thomson $T_e, n_e$ | inputs |

and writes one row in the FEC schema plus `efit_psirz`. Static tables (coil geometry, chord
positions) are copied from a released FEC row; they are given inputs, not labels. 3,738 shots
(numbers 23240–30471) were ingested, keeping all EFIT slices (ramp-up, limited, ramp-down).

**No-plasma shots.** 349 of the 3,738 shots carry a reduced EFIT++ product: 200 slices on a fixed
5 ms clock (0–995 ms or $-50$–945 ms), 60 instead of about 150 `efm` variables, no scalar or
X-point outputs, and $\max|I_p|\le46.8$ kA (median 9.1 kA). They are shots without a plasma,
spread over the whole shot range. 335 are in the QC keep list and 14 in the sealed 80.

`data/fair_mast/fetch_efit_labels.py` adds the scalar labels and the diverted flag. Its rule (any
finite `efm/xpoint{1,2}_rc`) reproduces the FEC frame selection on all three demo shots, and its
`q_95`, `betan` and edge-Thomson arrays match the FEC demo rows exactly (`tests/test_labels.py`).
For the three validation shots 28348, 28350 and 28351, the FAIR-MAST EFIT++ $\psi$, $q_{95}$ and
$\beta_N$ labels are bit-for-bit identical to the challenge organizers' release.
