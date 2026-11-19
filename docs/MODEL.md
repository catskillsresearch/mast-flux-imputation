# Model

The paper (`paper/MAST.md`, Model section) is the full description; this note is a summary with
pointers to the code.

## Output space

Each EFIT++ map $\mathbf y_t\in\mathbb R^{4225}$ ($65\times65$) is represented by 60 coefficients of
a PCA basis, $z_t=C(\mathbf y_t-\mu)$, $\hat{\mathbf y}_t=\mu+C^{\top}z_t$. $C$ and $\mu$ are fitted by
randomized SVD (seed 42, three power iterations) on at most 200,000 shot-balanced frames of the
FAIR-MAST training shots after the $|I_p|$ window (`models/m9/train.py`, `--build-cache`).

## Features (`models/m9/features.py`, `models/m9/network.py`)

| slice | content |
|---|---|
| 0–13 | 13 coil currents (EFPS, SOL, TF, P2–P6 lower/upper) and $I_p$, interpolated to the EFIT time |
| 14–275 | 131 core-Thomson $T_e$ and 131 $n_e$ channels (0 outside the Thomson clock) |
| 276–307 | 16 + 16 edge-Thomson channels; zeroed at prediction, raw in training (non-zero in 302 of 3,258 training shots) |

`augment_temporal` appends a centred window-9 moving average $a_t$ and $(a_{t+4}-a_{t-4})/8$ per
shot (924 features); features are standardised with training statistics and clipped to ±10. The
context reaches four EFIT frames ahead, so the model is an offline smoother.

## Network (`CoeffDualContext`, `build_dualctx`)

$$
h=\mathrm{Res}^{(4)}_{512}(\mathrm{ReLU}(W_h\tilde x_{[0,308)})),\quad
c=\mathrm{Res}^{(3)}_{512}(\mathrm{ReLU}(W_c\tilde x_{[308,924)})),\quad
h'=h+g\,Pc,\quad z=W_zh'+b_z,
$$

residual blocks of width 512 with dropout 0.05, gate $g$ initialised to 0. Five linear heads on
$h'$ predict standardised $q_{95}$, $\beta_N$, $R_{\rm axis}$, $Z_{\rm axis}$, $\ell_i$; the
$q_{95}$ and $\beta_N$ heads, de-standardised, are the scalar outputs. The scored axis and $\ell_i$
are derived from the predicted map.

## Differentiable read-outs (`experiments/e6.py`, `soft_feats`)

Used in training only. $\psi_N$ is formed with the EFIT++ axis and boundary flux; the interior is
$s=\sigma((1-\psi_N)/0.02)$ inside the EFIT++ LCFS dilated by two cells (vessel-envelope mask
where no EFIT++ LCFS exists). Column and row sub-grid extremes with soft-argmax (temperature
0.03 m) give $Z_{\max}$, $Z_{\min}$, $R_{\max}$, $R_{\min}$ and the $R$ of the highest and lowest
points, hence $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$; $V=2\pi\sum Rs\,\Delta A$; the axis is
the soft-argmin of $\psi_N$.

## Ensemble

Seeds 101, 202, 303, 404, 505; each contributes its rank-2 EMA state by validation objective
(epochs 103, 154, 106, 121, 86). The flux is the mean of the reconstructed maps, the scalars the
mean of the heads. Selection is described in `docs/TRAINING_REGIME.md`.
