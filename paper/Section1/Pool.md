# Training pool

1. **Split.** The 80 sealed holdout shots and the three validation shots are removed from the
   3,738 ingested shots, leaving a pool of 3,655.
2. **No-plasma shots** are removed from the pool: 335 shots.
3. **Quality control:** a shot is quarantined if $\max|\psi|$ exceeds the pool median by more than
   10 median absolute deviations, or if it is short, non-finite or constant: 3,280 shots are kept.
4. **$|I_p|$ window:** frames with $|I_p|\ge I_p^{\rm lo}=328.7293519$ kA are used. The threshold
   is the midpoint of a seed-42 2-means fit to all frame values of $|I_p|$ in the pool with the
   no-plasma shots included; without them the same fit gives 471.99 kA, so the value is fixed as a
   named constant. 3,258 shots contribute 211,534 frames, limited and diverted frames alike.
5. **Archive validation:** 10% of the contributing shots after a seed-42 shuffle, 325 shots, are
   held out of training. Its first 100 shots, `archval100`, carry the validation objective and the
   selection score of the calibration.

The principal-component basis and the feature statistics are fitted on the training shots only.
