# FEC data (`data/fec/`)

* `python data/fec/fetch.py --demo` → `demo/mast_shot_{28348,28350,28351}.parquet` (labelled,
  sha256-checked, from the starter kit's Git LFS).
* `python data/fec/fetch.py --public-test [--revision <commit>]` → `mast_public_test/` (1,206
  unlabelled rows, HF `Sophelio/fusion-equilibrium-challenge`, CC BY 4.0).

Everything here except this README, `fetch.py` and `.gitkeep` is git-ignored. Alternatively set
`FEC_DEMO_DIR` / `FEC_MAST_DIR`. Schema: see the top-level README §2.
