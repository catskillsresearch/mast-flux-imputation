# Relation to the training-log figure

The M9 training log reports a "sealed $R^2$" of about 0.378. That figure is a pixel $R^2$ pooled over
*all* 8,161 EFIT slices of the 80 sealed shots — ramp-up, limited and ramp-down phases included —
and it contains shot 27373, quarantined from training as a QC outlier. Recomputed with the shipped
ensemble it is 0.378687, and 0.733818 without shot 27373. On the FEC protocol (diverted slices only)
the same ensemble gives the 0.997771 above. The difference is the frame population, not a different
model.
