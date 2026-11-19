# Boundary and critical points

The viewer flags, per frame, the boundary and critical-point distances between the predicted and
the EFIT++ map. Their fold means are:

@import "../generated/model_route.md"

Table: Boundary and critical-point distances. Tip gap: vertical gap of the LCFS top or bottom of at least 50 mm; flips: frames where the predicted map is limited and EFIT++ diverted, or the reverse.

On the holdout the LCFS of the model lies within a mean Hausdorff distance of 45.5 mm of EFIT++,
with a 95th percentile of 102.3 mm. The largest boundary errors are at the top and bottom of the
LCFS: 32.4% of holdout frames have a tip gap of at least 50 mm. The O-point is placed within
11.7 mm and the X-point within 34.3 mm on average, and 3.6% of frames flip between limited and
diverted. The three validation shots have smaller distances on every row.
