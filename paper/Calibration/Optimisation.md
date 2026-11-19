# Optimisation and stopping

**Batches.** Each step takes 32 whole training shots and up to 128 frames per shot, in time order,
so that the per-shot curve distance and the frame differences are defined within the batch.

**Optimiser.** AdamW [@loshchilov2019] with learning rate $10^{-3}$, weight decay $10^{-4}$ and
gradient-norm clip 1.0; ReduceLROnPlateau on the validation objective (factor 0.5, patience 4,
threshold $10^{-3}$, minimum $2\times10^{-5}$). An exponential moving average (EMA) of the weights
with decay 0.998 per step is kept and evaluated. Every member is trained from scratch. PyTorch
[@paszke2019].

**Validation objective.** After each epoch the EMA weights are scored on `archval100` by

$$
J = \frac{\sum_k\omega_k\,\overline{\big(\mathrm{nRMSE}_{k,n}+\lambda_d\,\mathrm{nRMSE}'_{k,n}\big)}
 + \omega_{\rm LCFS}\,\overline{\big(1-\mathrm{IoU}_{\rm soft}\big)}_{\,n}}{\sum_k\omega_k+\omega_{\rm LCFS}},
$$

where $\mathrm{nRMSE}_{k,n}$ and $\mathrm{nRMSE}'_{k,n}$ are the square roots of the two per-shot terms
of $d_{k,n}$, and the LCFS term is the late-weighted per-shot mean.

**Spike guard.** If $J$ exceeds 1.2 times the best value so far, the weights return to the best EMA
state and the learning rate is halved.

**Stopping.** Training stops when $J$ has not improved by more than 0.2% for 20 epochs and is no
longer falling (less than 1% below its value five epochs earlier), with at least 30 and at most
300 epochs. The three EMA states with the lowest $J$ are kept for each member.
