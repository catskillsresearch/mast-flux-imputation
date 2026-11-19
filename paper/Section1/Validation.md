# Validation shots

Shots 28348, 28350 and 28351 are three consecutive MAST shots held out of the training pool, the
quality-control statistics, the principal-component basis, the feature statistics and checkpoint
selection. Their inputs and EFIT++ labels are downloaded from FAIR-MAST by the same ingest and label
scripts as the archive,

```
python data/fair_mast/ingest.py --keys 28348 28350 28351 \
    --out data/fair_mast/validation/parquet
python data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351 \
    --out data/fair_mast/validation/labels
```

and are scored on their 36, 42 and 37 diverted slices (115 frames). With only three shots, pooled
$R^2$ values on them are noisy; they are reported per shot alongside the holdout.
