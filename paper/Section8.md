# Future work: audit-guided training

During development the author compared imputed and EFIT++ boundaries by eye in the viewer
`app/fec_site`, frame by frame, and inspected the tracked time series for degeneracies. That does
not scale to the 3,738 ingested archive shots, let alone all of FAIR-MAST. The per-frame quantities,
time-series checks and failure tags below are implemented in `pipeline/audit.py` and produced the
evidence of Section 7.5; the ranked reports linked into the viewer, the use of the audit in training,
and the end-to-end workflow are planned and not implemented. Together they form the direct test of
the hypothesis of Section 7.3: an MSE-only model and a route-guided model, trained on the same
FAIR-MAST data and compared on the same sealed holdout.

- [What the viewer showed by eye](Section8/Viewer.md)
- [Per-frame audit quantities](Section8/Quantities.md)
- [Time-series checks](Section8/TimeSeries.md)
- [Failure classes and ranked reports](Section8/Classes.md)
- [Using the audit in training](Section8/Training.md)
- [Planned end-to-end automated workflow](Section8/Workflow.md)
