# Commands

The full chain runs from one `Makefile` with settings in `workflow/config.mk`:

```
bash scripts/run_xcheck.sh   # make all, as a systemd user unit
make download labels cache   # FAIR-MAST rows, labels, features
make prep route              # split, QC, |Ip| window, PCA basis
make features smoke          # diff vs an existing copy; current model on fresh rows
make sweep train select      # weight sweep, five seeds, selection
make score compare           # validation shots, holdout, comparison
make paper
```

Every step writes a stamp and its outputs atomically under `work/xcheck/`, so a finished step is
skipped and an interrupted run resumes. The data preparation rules are fixed in code:

* EFIT and Thomson clocks are converted from seconds to ms. The edge Thomson block is kept only
  when its clock reads as seconds (maximum below 10), the rule the training rows were built with;
  at prediction the 32 edge columns are set to zero.
* Magnetics are interpolated onto a fixed 0.2 ms grid from $-2000$ to $3999.8$ ms; currents and
  Thomson profiles are then interpolated to the EFIT times, Thomson being zero outside its clock.
* `efm/psirz` is stored on a doubled 129-column R grid; every other column is dropped to give
  $65\times65$ on the shot's own grid.
* Shots with exactly 200 EFIT slices on the fixed 5 ms clock carry no plasma and are removed;
  training frames need $|I_p|\ge 328.73$ kA; only diverted frames are scored.

The individual commands are:

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
