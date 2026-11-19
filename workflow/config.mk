# One config file for the end-to-end workflow (Makefile at the repo root).
# Every output goes under $(WORK) (git-ignored). Nothing is ever written to the lab drive.

WORK            ?= $(CURDIR)/work/xcheck
PYTHON          ?= $(CURDIR)/.venv/bin/python
# 3,738 FAIR-MAST shots with efm + ayc
SHOT_POOL       ?= data/fair_mast/mast_pool3.txt
# shots per resumable download chunk
CHUNK           ?= 100
# parallel download chunks
DL_JOBS         ?= 12
LABEL_WORKERS   ?= 16
CACHE_WORKERS   ?= 8
EVAL_WORKERS    ?= 8
# training | overlap (data/fair_mast/ingest.py)
EDGE_CLOCK      ?= training
# refuse to start below this much free disk
FREE_GB_MIN     ?= 60

# Model selection gate (archval100 S) used by the select step in place of an earlier run.
SELECT_GATE_S   ?= 0.934308

# Reference (current model) and pass/fail rule for the smoke check and the compare step.
REF_EXP         ?= $(CURDIR)/outputs/experiments/E6
REF_AUDIT       ?= $(CURDIR)/outputs/audit
LAB_CACHE       ?= /media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/cache/mast_features_v1
LAB_PARQUET     ?= /media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/datasets/fair_mast_train
# fail if holdout S is more than this below the reference
S_DROP_MAX      ?= 0.01
# fail if any score term falls more than this below the reference
TERM_DROP_MAX   ?= 0.10
