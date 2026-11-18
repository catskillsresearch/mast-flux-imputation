# How OSF was calibrated

No fitting code exists for OSF; its 38 calibrated values (Section 3.9) are hand-set constants. Per
the calibration notes of the source repository they were chosen by

* inspecting physics end-members (ohmic start, solenoid swing, diverted flat-top) on the three
  labelled starter-kit shots, with $R^2$ values on those shots quoted to justify shape choices; and
* five aggregate-score probes of the Codabench leaderboard on `mast_public_test`.

The demo shots are therefore *not* held out for OSF, and the public-test fold has been used,
in aggregate, for tuning. The sealed archive holdout is label-disjoint from all of this, but OSF's
global factors were not chosen blind to MAST data. We report the demo-shot scores separately and
for reference only.
