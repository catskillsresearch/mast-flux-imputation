# FAIR-MAST archive (`data/fair_mast/`)

| file | role |
|---|---|
| `ingest.py` | FAIR-MAST Zarr (`efm`, `amc`, `ayc`, `aye`) → FEC-schema parquet + `efit_psirz` (`parquet/`, ignored) |
| `fetch_efit_labels.py` | efm `q_95`, `betan`, X-points, diverted flag + edge Thomson → `labels/*.npz` (ignored) |
| `make_holdout.py` | recomputes the sealed-80 split and the disjointness checks → `holdout_seed42.json` |
| `holdout_seed42.json` | committed manifest: protocol, 80 sealed keys, checks |
| `mast_pool3.txt` | candidate shot pool used by the ingest |
| `mast_qc_keep.txt`, `mast_qc_v1_index.json`, `mast_qc_index_meta.json` | M9 training QC manifests (3,622 kept shots) |

Source: `s3://mast/level1/shots/<N>.zarr` at `https://s3.echo.stfc.ac.uk` (anonymous), CC BY-SA 4.0.
Requires the `fetch` extra. Alternatively set `FAIR_MAST_DIR` / `FAIR_MAST_LABELS_DIR`.
