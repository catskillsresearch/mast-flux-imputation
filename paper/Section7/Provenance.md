# Data provenance

1. **Training inputs and labels** are FAIR-MAST shots read from `s3://mast/level1/shots` by
   `data/fair_mast/ingest.py`: currents (`amc`), Thomson (`ayc`, `aye`) and EFIT++ (`efm`). The
   feature cache is rebuilt exactly from these rows, and every model input comes from them.
2. **Scalar labels** ($q_{95}$, $\beta_N$, axis, $\ell_i$, diverted flag) are read from FAIR-MAST
   `efm` by `data/fair_mast/fetch_efit_labels.py`.
3. **PCA basis, mean and feature statistics** are fitted on the training shots.
4. **Archive validation and the sealed holdout** are FAIR-MAST shots processed by the same ingest
   and label scripts.
5. **The three validation shots** are downloaded from FAIR-MAST by the same scripts into
   `data/fair_mast/validation/`.
6. **Scoring code and vessel-envelope mask** (`fec_scoring/`, `mast_envelope.npz`) come from the FEC
   MAST task; the mask is also used by the soft read-outs where no EFIT++ LCFS is available.
7. **No magnetic sensor** is a model input; the EFIT++ labels were reconstructed with magnetic
   sensors.

The same list, with file-level detail, is in `DATA_PROVENANCE.md` of the repository.
