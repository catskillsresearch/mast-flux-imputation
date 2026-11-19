# Commands

```
python data/fair_mast/ingest.py \
    --pool data/fair_mast/mast_pool3.txt --shots 5000
python data/fair_mast/make_holdout.py     # sealed-80 + checks
python data/fair_mast/fetch_efit_labels.py \
    --keys-json data/fair_mast/holdout_seed42.json
python data/fair_mast/ingest.py --keys 28348 28350 28351 \
    --out data/fair_mast/validation/parquet
python data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351 \
    --out data/fair_mast/validation/labels
python -m models.m9.train --build-cache   # features, split, PCA basis
bash scripts/run_e6.sh                    # train, select, predict, score
python scripts/make_compare_preds.py demo holdout  # viewer predictions
python -m pipeline.score outputs/compare/demo_e6.npz
python -m pipeline.audit outputs/compare/demo_e6.npz \
    --tag compare_demo_e6                 # boundary and curve distances
python scripts/make_model_report.py       # tables and figures
python -m app.fec_site --port 8770        # viewer
```

Predictions are `.npz` files in the template layout with keys `shot_XXXX_psirz` $(T,65,65)$,
`shot_XXXX_q95` and `shot_XXXX_betaN` $(T,)$, plus a JSON sidecar giving the shot order. Models only
see `inputs_only(row)`, which removes every `efit_*` label column. Raw data, trained weights and
outputs are not in the repository; paths may point at an existing data copy through
`FAIR_MAST_DIR`, `FAIR_MAST_LABELS_DIR` or `CATSKILLS_FUSION_LAB`.
