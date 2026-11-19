# How OSF was calibrated

No fitting code exists for OSF's solver; its 38 calibrated values (Section 3.9) are hand-set
constants. They were chosen by

* inspecting physics end-members (ohmic start, solenoid swing, diverted flat-top) on the three
  labelled starter-kit shots, with $R^2$ values on those shots used to justify shape choices; and
* a small number of aggregate scores on the FEC public-test fold `mast_public_test`.

The three validation shots are therefore *not* held out for OSF, and the public-test fold has been
used, in aggregate, for tuning. The sealed archive holdout is label-disjoint from all of this. The
$q_{95}$ and $\beta_N$ read-out maps of the hybrid are fitted on FAIR-MAST training shots only
(experiment E5, Section 10.6).
