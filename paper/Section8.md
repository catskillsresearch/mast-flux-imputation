# Future work: an automated failure-mode audit

This section is a proposal; none of it has been implemented. During development the author compared
imputed and EFIT++ boundaries by eye in the viewer `app/fec_site`, frame by frame, and inspected the
tracked time series for degeneracies. That does not scale to the 3,738 ingested archive shots, let alone all of FAIR-MAST. We propose to
make the same inspection systematic.

- [What the viewer showed by eye](Section8/Viewer.md)
- [Per-frame audit quantities](Section8/Quantities.md)
- [Time-series checks](Section8/TimeSeries.md)
- [Failure classes and ranked reports](Section8/Classes.md)
- [Using the audit in training](Section8/Training.md)
