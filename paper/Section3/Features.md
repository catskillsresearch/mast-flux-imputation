# Features and temporal context

For each EFIT frame $t$ a raw vector $x_t\in\mathbb R^{308}$ is built:

| slice | content |
|---|---|
| 0–13 | 13 coil currents (EFPS, SOL, TF, P2–P6 lower/upper) and $I_p$, linearly interpolated to $t$ |
| 14–275 | 131 core-Thomson $T_e$ and 131 $n_e$ channels, interpolated in time (0 outside the Thomson clock) |
| 276–307 | 16 + 16 edge-Thomson channels, set to zero |

No spatial remapping of the Thomson channels is done; channel index stands in for radius. A per-shot
temporal augmentation with window 9 appends the centred moving average $a_t$ (clamped at the shot
edges) and its smoothed central difference $(a_{t+4}-a_{t-4})/8$, giving
$A(x)_t=[x_t\,|\,a_t\,|\,\dot a_t]\in\mathbb R^{924}$. Features are standardised with the training
mean and standard deviation and clipped to $\pm10$:
$\tilde x_t=\mathrm{clip}_{\pm10}\big((A(x)_t-m)/\sigma\big)$.

Since the context uses EFIT frames up to four steps ahead, the model is a smoother over the shot,
not a causal real-time estimator.
