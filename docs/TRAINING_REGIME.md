# Training regime

How the model is trained and selected (`experiments/e6.py`, `experiments/driver_e6.py`, launched
by `bash scripts/run_e6.sh`). The sealed holdout and the three validation shots are not used.

## Data

* Pool: the 3,738 ingested FAIR-MAST shots minus the three validation shots and the 80 sealed
  holdout shots (seed 42): 3,655 shots.
* The 335 no-plasma shots of the pool are removed; quality control keeps 3,280 shots.
* $|I_p|$ window fixed at 328.7293519 kA: 3,258 shots contribute 211,534 frames.
* Archive validation: 10% of the contributing shots after a seed-42 shuffle (325 shots).
  `archval100`, its first 100 shots, carries the validation objective $J$ and the
  selection score $S$.
* One 60-mode PCA basis and one set of feature statistics, fitted on the training shots.

## Loss

$$
\mathcal L=\overline{\|\hat z-z\|^2}+w_{cd}\mathcal L_{cd}+w_{\rm LCFS}(1-\mathrm{IoU}_{\rm soft})
+w_b\mathcal L_b+w_{\rm tip}\mathcal L_{\rm tip}+w_c\mathcal L_c+w_t\mathcal L_t+w_s\,\mathrm{Huber}
$$

| term | definition | weight |
|---|---|---|
| $\mathcal L_{cd}$ | per-shot range-normalised squared error plus 0.5 × the same on frame differences, for $q_{95}$, $\beta_N$ (heads) and $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$, $V$, $R_{\rm axis}$, $Z_{\rm axis}$ (soft read-outs); range floored at 0.25 global std | $w_{cd}=0.01$ |
| soft LCFS | 1 − soft IoU of the soft interior and the EFIT++ LCFS interior | $w_{cd}/3$ |
| $\mathcal L_b$ | pixel error in $\lvert\psi_N-1\rvert<0.1$ of the EFIT++ LCFS | 3 |
| $\mathcal L_{\rm tip}$ | the same in the top and bottom 15% of the LCFS height | 10 |
| $\mathcal L_c$ | $\lvert\nabla\hat\psi\rvert^2$ at EFIT++ O/X points and X-point flux mismatch | $10^{-3}$ |
| $\mathcal L_t$ | frame-to-frame coefficient differences | 1 |
| Huber | five standardised scalar heads | $3\times10^{-3}$ |

Late weight $1+\alpha\,\mathrm{clip}((t_{\rm frac}-0.8)/0.2,0,1)$ with $\alpha=4$ for $\beta_N$ and 2
otherwise; series weight $\omega=2$ for $\beta_N$, $\delta_{\rm top}$, $\delta_{\rm bot}$ and 1
otherwise.

## Optimisation and stopping

* 32 whole shots per step, up to 128 frames per shot in time order.
* AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, gradient clip 1.0; ReduceLROnPlateau
  (factor 0.5, patience 4, threshold $10^{-3}$, minimum $2\times10^{-5}$); EMA of the weights,
  decay 0.998 per step. Every member is trained from scratch.
* Validation objective $J$ on `archval100`: $\omega$-weighted per-shot nRMSE plus 0.5 × the
  derivative nRMSE, late-weighted, plus the late-weighted soft-LCFS term ($\alpha=2$,
  $\omega_{\rm LCFS}=1$).
* Spike guard: if $J>1.2\times$ best, return to the best EMA state and halve the learning rate.
* Stop when $J$ has not improved by more than 0.2% for 20 epochs and is less than 1% below its
  value five epochs earlier; 30 to 300 epochs. Keep the three best EMA states per member.

## Stages

| stage | what it does |
|---|---|
| `truth` | soft read-outs of every EFIT++ training frame (cached) |
| `sweep` | $w_{cd}\in\{0.01,0.03,0.1,0.3\}$ on seed 101, 40 epochs; lowest $J$ among settings whose best coefficient MSE is within 10% of the lowest (0.01) |
| `m101` … `m505` | five members with the chosen weights |
| `select` | for $k=1,2,3$, the ensemble of the $k$-th best states; the first with archive-validation $S\ge0.934308$ is used (rank 2) |
| `final` | holdout prediction, scoring and audit |

The tables of the sweep, the member records and the selection are generated into
`paper/generated/model_{sweep,training,selection}.md` by `scripts/make_model_report.py`.
