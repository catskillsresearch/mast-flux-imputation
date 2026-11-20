# End-to-end workflow: FAIR-MAST download -> features -> training -> scoring -> comparison.
# Settings: workflow/config.mk. Outputs: $(WORK) (git-ignored). Each step leaves a stamp in
# $(WORK)/stamps; a finished step is skipped on the next `make`. Steps write atomically
# (temp file + rename), so an interrupted run resumes cleanly.
#
#   make all          # everything below, in order
#   make status       # $(WORK)/STATUS.json
#   make paper        # rebuild paper/MAST.pdf from the tracked sources
include workflow/config.mk

STAMP := $(WORK)/stamps
ST    := $(PYTHON) scripts/xcheck.py status
DRV   := $(PYTHON) -m experiments.driver_e6 --stage
SEEDS := 101 202 303 404 505

export FAIR_MAST_DIR        := $(WORK)/parquet
export FAIR_MAST_LABELS_DIR := $(WORK)/labels
export MFI_FEATURE_CACHE    := $(WORK)/cache
export MFI_OUTPUT_DIR       := $(WORK)/outputs
export MFI_EXP_DIR          := $(WORK)/outputs/experiments
export MFI_SPLITS_JSON      := $(WORK)/experiment_splits_seed42.json
export MFI_VALIDATION_DIR   := $(WORK)/validation
export MFI_SELECT_GATE_S    := $(SELECT_GATE_S)
export MFI_WORK             := $(WORK)
export TMPDIR               ?= /tmp
export PYTHONUNBUFFERED     := 1
unexport CATSKILLS_FUSION_LAB

N_SHOTS  := $(shell wc -w < $(SHOT_POOL))
N_CHUNKS := $(shell echo $$(( ($(N_SHOTS) + $(CHUNK) - 1) / $(CHUNK) )))
CHUNKS   := $(shell seq 0 $$(( $(N_CHUNKS) - 1 )))

.PHONY: all diskcheck shotlist download labels validation cache noplasma prep route features \
        smoke truth sweep train select predict score report compare paper status clean-stamps
.DELETE_ON_ERROR:

all: compare

$(STAMP)/.dir:
	@mkdir -p $(STAMP) $(WORK)/logs && touch $@

# --- 0. disk space (refuses to start when the work disk is short) -----------------------------
diskcheck: $(STAMP)/.dir
	@$(PYTHON) scripts/xcheck.py diskcheck --min-gb $(FREE_GB_MIN)

# --- 1. shot list ------------------------------------------------------------------------------
shotlist: $(STAMP)/shotlist
$(STAMP)/shotlist: $(SHOT_POOL) | diskcheck
	$(ST) shotlist running
	$(PYTHON) scripts/xcheck.py shotlist --pool $(SHOT_POOL)
	$(ST) shotlist done && touch $@

# --- 2. download (per-shot parquet, resumable, parallel chunks) --------------------------------
download: $(STAMP)/download
$(STAMP)/download: $(STAMP)/shotlist
	$(ST) download running
	$(MAKE) --no-print-directory -j$(DL_JOBS) $(addprefix $(STAMP)/dl_,$(CHUNKS))
	$(PYTHON) scripts/xcheck.py count --what parquet
	$(ST) download done && touch $@
$(STAMP)/dl_%:
	$(PYTHON) data/fair_mast/ingest.py --pool $(SHOT_POOL) --start $$(( $* * $(CHUNK) )) \
	  --shots $(CHUNK) --edge-clock $(EDGE_CLOCK) --out $(FAIR_MAST_DIR) > $(WORK)/logs/dl_$*.log 2>&1
	@touch $@

labels: $(STAMP)/labels
$(STAMP)/labels: $(STAMP)/download
	$(ST) labels running
	$(PYTHON) scripts/xcheck.py keys-json --out $(WORK)/keys_all.json
	$(PYTHON) data/fair_mast/fetch_efit_labels.py --keys-json $(WORK)/keys_all.json \
	  --workers $(LABEL_WORKERS) --out $(FAIR_MAST_LABELS_DIR) > $(WORK)/logs/labels.log 2>&1
	$(ST) labels done && touch $@

validation: $(STAMP)/validation
$(STAMP)/validation: $(STAMP)/shotlist
	$(ST) validation running
	$(PYTHON) data/fair_mast/ingest.py --keys 28348 28350 28351 --edge-clock $(EDGE_CLOCK) \
	  --out $(MFI_VALIDATION_DIR)/parquet > $(WORK)/logs/validation.log 2>&1
	$(PYTHON) data/fair_mast/fetch_efit_labels.py --shots 28348 28350 28351 \
	  --out $(MFI_VALIDATION_DIR)/labels >> $(WORK)/logs/validation.log 2>&1
	$(ST) validation done && touch $@

# --- 3. features (window-free 308-column rows + 65x65 psi targets per shot) --------------------
cache: $(STAMP)/cache
$(STAMP)/cache: $(STAMP)/download
	$(ST) cache running
	$(PYTHON) -m models.m9.train --build-cache --feature-cache $(MFI_FEATURE_CACHE) \
	  --workers $(CACHE_WORKERS) > $(WORK)/logs/cache.log 2>&1
	$(ST) cache done && touch $@

# no-plasma shots (exactly 200 EFIT slices on the fixed 5 ms clock) -> outputs/compat/no_plasma.json
noplasma: $(STAMP)/noplasma
$(STAMP)/noplasma: $(STAMP)/cache
	$(ST) noplasma running
	$(PYTHON) scripts/scan_no_plasma_shots.py --feature-cache $(MFI_FEATURE_CACHE) > $(WORK)/logs/noplasma.log 2>&1
	$(ST) noplasma done && touch $@

# --- 4. split, QC, |Ip| window, PCA basis, feature stats --------------------------------------
prep: $(STAMP)/prep
$(STAMP)/prep: $(STAMP)/noplasma $(STAMP)/labels
	$(ST) prep running
	$(PYTHON) -m experiments.prep > $(WORK)/logs/prep.log 2>&1
	$(PYTHON) data/fair_mast/make_holdout.py --ingest-dir $(FAIR_MAST_DIR) --out $(WORK)/holdout_seed42.json >> $(WORK)/logs/prep.log 2>&1
	$(ST) prep done && touch $@

route: $(STAMP)/route
$(STAMP)/route: $(STAMP)/prep
	$(ST) route running
	$(PYTHON) -m experiments.route_labels > $(WORK)/logs/route.log 2>&1
	$(ST) route done && touch $@

# --- 5. checks before training: feature diff vs the lab cache, current model on fresh rows -----
features: $(STAMP)/features
$(STAMP)/features: $(STAMP)/prep
	$(ST) features running
	$(PYTHON) scripts/xcheck.py features --lab-cache $(LAB_CACHE) --lab-parquet $(LAB_PARQUET)
	$(ST) features done && touch $@

smoke: $(STAMP)/smoke
$(STAMP)/smoke: $(STAMP)/features $(STAMP)/validation
	$(ST) smoke running
	$(PYTHON) scripts/xcheck.py smoke --ref-exp $(REF_EXP) --ref-audit $(REF_AUDIT) \
	  --s-drop-max $(S_DROP_MAX) --term-drop-max $(TERM_DROP_MAX)
	$(ST) smoke done && touch $@

# --- 6. training: truth tables, weight sweep, 5 seeds, checkpoint selection ---------------------
truth: $(STAMP)/truth
$(STAMP)/truth: $(STAMP)/route $(STAMP)/smoke
	$(ST) truth running && $(DRV) truth && $(ST) truth done && touch $@

sweep: $(STAMP)/sweep
$(STAMP)/sweep: $(STAMP)/truth
	$(ST) sweep running && $(DRV) sweep && $(ST) sweep done && touch $@

train: $(addprefix $(STAMP)/m,$(SEEDS))
$(STAMP)/m101: $(STAMP)/sweep
$(STAMP)/m202: $(STAMP)/m101
$(STAMP)/m303: $(STAMP)/m202
$(STAMP)/m404: $(STAMP)/m303
$(STAMP)/m505: $(STAMP)/m404
$(STAMP)/m%:
	$(ST) m$* running && $(DRV) m$* && $(ST) m$* done && touch $@

select: $(STAMP)/select
$(STAMP)/select: $(addprefix $(STAMP)/m,$(SEEDS))
	$(ST) select running && $(DRV) select && $(ST) select done && touch $@

# --- 7. predictions and scoring (validation shots, sealed holdout) -----------------------------
predict score: $(STAMP)/score
$(STAMP)/score: $(STAMP)/select
	$(ST) score running && $(DRV) final && $(ST) score done && touch $@

# --- 8. report and comparison with the current model -------------------------------------------
compare report: $(STAMP)/compare
$(STAMP)/compare: $(STAMP)/score
	$(ST) compare running
	$(PYTHON) scripts/xcheck.py compare --ref-exp $(REF_EXP) --ref-audit $(REF_AUDIT) \
	  --s-drop-max $(S_DROP_MAX) --term-drop-max $(TERM_DROP_MAX)
	$(ST) compare done && touch $@

# --- paper (tracked sources -> paper/MAST.pdf) --------------------------------------------------
paper:
	bash scripts/build_paper_pdf.sh

status:
	@cat $(WORK)/STATUS.json 2>/dev/null || echo "no $(WORK)/STATUS.json"

clean-stamps:
	rm -rf $(STAMP)
