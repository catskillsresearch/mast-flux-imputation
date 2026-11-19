# Planned end-to-end automated workflow

*Planned; not implemented.* The repository currently provides the individual steps as separate
commands (Section 5.4). The goal is a single parameterised, controllable workflow that rebuilds both
models from public data with no manual intervention:

1. **Download.** FEC MAST public-test fold and the three labelled starter-kit shots from Hugging Face
   [@fec2026dataset; @fec2026starter]; FAIR-MAST shots from the public S3 archive
   [@jackson2024fairmast], each file checksummed.
2. **Ingest** FAIR-MAST into the FEC parquet schema, using the same clocks, resampling, chord radii
   and edge-Thomson handling as the FEC rows (see the compatibility notes below), and fetch the
   EFIT++ scalar labels.
3. **QC and split.** The quality fence, the $|I_p|$ window and the sealed holdout, all
   recomputed with recorded seeds; the disjointness checks run and fail the workflow if violated.
4. **Calibrate OSF** from scratch: the calibrated parameter group (Section 3.9) fitted by an explicit
   optimiser on training shots only, replacing the hand-set values.
5. **Train M9** from scratch: PCA basis, network and ensemble members.
6. **Validate and gate.** Validation on the three FEC demo shots — provided the compatibility
   conditions below are met — together with the archive validation split, and the failure-mode audit
   of this section as the acceptance gate for checkpoints and calibrations.
7. **Score the sealed holdout once**, at the end, and write the tables and figures of this paper.

Every parameter (data sources and shot lists, seeds, QC thresholds, PCA rank, network sizes,
optimiser settings, calibration bounds, audit thresholds) would come from one versioned
configuration file, each stage would be cached and resumable, and every run would record its
configuration, data checksums and code version alongside its outputs.

**Demo shots as validation set.** A column-by-column comparison of the FAIR-MAST ingest rows of
shots 28348, 28350 and 28351 with the FEC demo rows shows: the EFIT++ $\psi$ on the 36, 42 and 37
diverted frames is bit-identical; the M9 input features at those frames agree (currents within
0.001 kA, core Thomson to relative $5\times10^{-5}$); and the three shots are excluded from the
training pool, the QC keep list and the PCA basis. They differ in edge Thomson (present in the demo
rows, absent from the ingest rows), in the magnetics clock (a 0.1 ms offset of the raw 5 kHz
series, which changes $I_p$ by up to 63 kA in fast transients but not at the EFIT frames), in the
core chord-radius table (131 versus 130 entries), and in the ingest carrying all EFIT slices but no
scalar labels. With edge Thomson and the clocks harmonised, the demo shots are compatible as a
validation set. They remain a very small one (three consecutive shots, 115 frames) and were already
used for OSF tuning, so they should complement, not replace, a larger
archive validation split, and they cannot be reported as held-out results.
