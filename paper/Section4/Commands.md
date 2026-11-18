# Commands

```
python data/fec/fetch.py --demo           # 3 labelled shots
python data/fec/fetch.py --public-test    # 1,206 public-test shots
python data/fair_mast/ingest.py \
    --pool data/fair_mast/mast_pool3.txt --shots 5000
python data/fair_mast/make_holdout.py     # sealed-80 + checks
python data/fair_mast/fetch_efit_labels.py \
    --keys-json data/fair_mast/holdout_seed42.json
python -m models.m9.train --build-cache
python -m models.m9.train --member all
python -m pipeline.predict --split holdout --mode osf
python -m pipeline.predict --split holdout --mode hybrid
python -m pipeline.score outputs/holdout_hybrid.npz
python -m app.fec_site --port 8770
```

Predictions are FEC-format `.npz` files with keys `shot_XXXX_psirz` $(T,65,65)$, `shot_XXXX_q95`
and `shot_XXXX_betaN` $(T,)$, plus a JSON sidecar giving the shot order. Models only see
`inputs_only(row)`, which removes every `efit_*` label column. Paths may point at an existing data
copy through `FEC_MAST_DIR`, `FEC_DEMO_DIR`, `FAIR_MAST_DIR`, `FAIR_MAST_LABELS_DIR`,
`M9_WEIGHTS_DIR` or `CATSKILLS_FUSION_LAB`.
