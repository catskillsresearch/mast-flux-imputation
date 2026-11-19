# Training

The base recipe (`models/m9/train.py` with `--seed 42 --epochs 50 --qc --shot-balanced --mask-edge
--win`; the experiments of Section 10.6 call it through `experiments/m9x.py`) is:

1. **Split** as in Section 2.4: 80 sealed shots, pool of 3,655.
2. **No-plasma shots** (Section 2.2) removed from the pool: 335 shots.
3. **QC:** quarantine a shot if $\max|\psi|$ exceeds the pool median by more than 10 median
   absolute deviations, or if it is short, non-finite or constant: 3,280 kept.
4. **$|I_p|$ window:** frames with $|I_p|\ge I_p^{\rm lo}=328.7293519$ kA are used. The threshold
   is the midpoint of a seed-42 2-means fit to all frame values of $|I_p|$ in the pool before the
   no-plasma shots are removed; without them the same fit gives 471.99 kA, so it is fixed as a
   named constant. 3,258 shots contribute 211,534 frames. Limited and diverted frames are both
   included.
5. **PCA** with $K=60$ (Section 4.2), fitted once on the training shots.
6. **Validation:** 10% of the contributing shots (seed-42 shuffle), 325 shots.
7. **Optimisation:** AdamW [@loshchilov2019] (learning rate $10^{-3}$, weight decay $10^{-4}$),
   gradient-norm clip 1.0, batches of 2048 frames drawn shot-balanced (uniform shot, then uniform
   frame), $\lceil N_{\rm train}/2048\rceil$ steps per epoch, 50 epochs, keeping the epoch with the
   lowest validation coefficient MSE (47, 46 and 49 for members 101, 202 and 303 of the MSE-only
   run E1). PyTorch [@paszke2019].

The route-guided runs of Section 10.6 keep this data, split, basis and optimiser and change only
the loss and the checkpoint rule.
