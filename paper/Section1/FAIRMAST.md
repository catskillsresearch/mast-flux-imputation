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
were ingested. For the same shot and time, FAIR-MAST and FEC flux maps are bit-identical; the
populations differ only because the archive keeps all slices (ramp-up, limited, ramp-down).

`data/fair_mast/fetch_efit_labels.py` adds the scalar labels and the diverted flag. Its rule (any
finite `efm/xpoint{1,2}_rc`) reproduces the FEC frame selection on all three demo shots, and its
`q_95`, `betan` and edge-Thomson arrays match the FEC demo rows exactly (`tests/test_labels.py`).
