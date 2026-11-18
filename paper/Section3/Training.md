# Training

The recipe (`models/m9/train.py`, a port of the original with flags
`--seed 42 --epochs 50 --qc --shot-balanced --mask-edge --win`) is:

1. **Split** as in Section 2.4: 80 sealed shots, pool of 3,655.
2. **QC:** quarantine a shot if $\max|\psi|$ exceeds the pool median by more than 10 median
   absolute deviations, or if it is short, non-finite or constant: 3,622 kept.
3. **$|I_p|$ window:** 2-means on all keep-shot frame values of $|I_p|$ (seed 42); frames with
   $|I_p|$ at or above the midpoint of the two centres, $I_p^{\rm lo}=328.7293519015368$ kA, are
   used. 3,265 shots contribute frames; the other 357 are the 335 no-plasma shots of Section 2.2
and 22 shots whose $|I_p|$ stays below the threshold. Limited and diverted frames are both
included. The no-plasma shots contribute no training frame and do not enter the PCA basis, but
their near-zero $|I_p|$ frames enter the 2-means fit: without them the same fit gives
$I_p^{\rm lo}=471.99$ kA.
4. **PCA** with $K=60$ (Section 4.2).
5. **Validation:** 10% of the windowed shots (seed-42 shuffle).
6. **Optimisation:** AdamW [@loshchilov2019] (learning rate $10^{-3}$, weight decay $10^{-4}$),
   gradient-norm clip 1.0, batches of 2048 frames drawn shot-balanced (uniform shot, then uniform
   frame), $\lceil N_{\rm train}/2048\rceil$ steps per epoch, 50 epochs, keeping the epoch with the
   lowest validation MSE (46, 49 and 46 for the shipped members). PyTorch [@paszke2019].

During extraction the split, QC keep list and $I_p^{\rm lo}$ were reproduced exactly and a one-epoch
smoke test ran, but a full retrain was not carried out; the shipped weights are verified by
sha256 instead (Section 9).
