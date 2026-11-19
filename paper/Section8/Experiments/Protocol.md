# Protocol

The rules below were fixed in `docs/TRAINING_REGIME.md` before any experiment was run; the code is
`experiments/`, and one command, `scripts/run_experiments.sh`, runs every stage in order and
resumably.

* **Data.** The sealed-80 split of Section 2 is untouched. The training pool is the remaining
  archive minus the 335 no-plasma shots of the pool; QC is recomputed; the $|I_p|$ window is fixed
  at 328.7293519 kA; one 60-mode PCA basis is fitted once and shared by every experiment and
  member. This leaves 3,258 contributing shots and 211,534 frames (Section 4).
* **Splits.** Archive validation is the first 10% of the contributing shots after a seed-42
  shuffle (325 shots); `archval100` and `select30` are its first 100 and 30 shots; `osftrain200` is
  200 training shots (seed 43). None of these validation shots is used to fit any model. The three
  FEC validation shots are never trained on. Archive shots are evaluated with the FEC protocol cut
  (diverted EFIT frames only).
* **Route score.** $\mathcal R$ is the mean, over the tip-gap rate, the p95 and mean Hausdorff
  distances, the relative LCFS area mismatch, the X-point and O-point offsets and the topology-flip
  rate (plus 0.005), of the ratio of a model's value to a reference model's value on the same
  shots. $\mathcal R<1$ is better than the reference. Sweeps and checkpoint choices on `select30`
  use a fixed M9 ensemble trained with coefficient MSE as reference; the tables in this section
  report $\mathcal R$ relative to E1.
* **E1, MSE only.** The base recipe of Section 4.5: coefficient MSE, per member the epoch with the
  lowest archive-validation coefficient MSE, members 101/202/303, 50 epochs.
* **E2.** Coefficient MSE plus a boundary band (pixel error inside $|\psi_N-1|<0.1$ of the EFIT
  LCFS, weight $w_b$) and a soft-LCFS area term ($1-$soft IoU, weight $w_a$). A sweep on member 101
  (10 epochs) picks $(w_b,w_a)$ by the lowest $\mathcal R$ on `select30`; then three members of 50
  epochs; per member the checkpoint (epochs 19, 29, 39, 44, 49 or best MSE) with the lowest
  $\mathcal R$ on `select30`.
* **E3.** E2 plus the network's scalar heads trained with a Huber loss on standardised EFIT++
  $q_{95}$, $\beta_N$, $R_{\rm axis}$, $Z_{\rm axis}$, $\ell_i$ (weight $w_s$); the heads supply
  $q_{95}$ and $\beta_N$.
* **E4.** E3 plus critical-point terms ($|\nabla\hat\psi|^2$ at the EFIT O- and X-points and the
  X-point flux mismatch, weight $w_c$) and a temporal term on frame-to-frame coefficient
  differences (weight $w_t$).
* **E5.** OSF read-out maps from FAIR-MAST only: OSF is solved on `osftrain200` and affine maps of
  $q_{95}$ and $\beta_N$ (clipped to $[2,25]$ and $[0.2,5]$) are fitted by Nelder–Mead to the
  EFIT++ labels. E1 and E2 take $q_{95}$, $\beta_N$ from OSF through these maps (Section 5.1).
* **Final pick.** Among E2–E4, the lowest $\mathcal R$ on `archval100`; ties to four decimals
  broken by validation-shot $S$.
* **Holdout, once.** After the pick, E1, the pick and OSF with the E5 maps are scored once on the
  sealed holdout.
