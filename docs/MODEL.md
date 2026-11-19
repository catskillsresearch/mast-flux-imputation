# Model details

Symbols follow `models/osf/calibration.py`; the code is authoritative.

## OSF (`mast_12_osf_0.0.12`)

**Clocks** (measured currents at each EFIT time, `walk_on_measured=True`):
$w=(1-\min(\bar I_{P2}/(|I_p| r_\star),1))^2$, $b=\mathrm{clip}(I_{\rm sol}/I_\star,0,1)$,
$r_\star=0.34$, $I_\star=16.3$ kA.

**Vacuum field.** $\psi_{\rm vac}=\sum_c\lambda_c I_c\,\Psi_c$ with unit maps $\Psi_c$ from the
812-row coil table (3 × 3 sub-filaments per rectangle, elliptic-integral Green's function).
Multipliers: P2 pair $\lambda_V=1.8-b$, P2L extra $\times(1+0.105(1-b))$, P4 pair $1.15$, P5 pair
$\times(1-0.18(1-b))$, solenoid $\times(0.5+0.1b)$; P3, P6 unscaled; TF, EFPS excluded.

**Plasma current.** Support: ellipse (`USE_VACUUM_XD=False`), $R_0=0.82+0.10w-0.02b$,
$Z_0=0.009-0.004b$, $a_R=0.50+0.08w$, $a_Z=a_R+0.08+0.42b+0.18(1-b)$; $\tau=\rho^2$ inside.
$j=R\,p'(\tau)+FF'(\tau)/(\mu_0R)$ with $p'=\langle p\rangle(1-\tau)/(\Delta\psi\,s_V)$,
$FF'=-\bar\gamma(1-\tau)$, $\bar\gamma=-1$, then rescaled so $\int j\,dA=k I_p$ with
$k=0.70+0.30w+0.15b+0.20(1-b)$. $\langle p\rangle$ from a first-guess $\beta_N$ (`beta_guess="osf_0_0_1"`).
$\psi=\psi_{\rm vac}+G\,j$ (no wall currents). Picard: ≤ 20 passes, stop when
$\max|\delta\psi|<10^{-4}$ and the relative changes of $W$, $\Delta\psi$, $s_V$ are $<10^{-3}$.

**LCFS / axis.** O-point in $R\in[0.65,1.05]$, $|Z|\le0.5$; compact closed contour gates
($\kappa\le2.6$, $Z$-span $\le2.8$ m, $R_{\rm out}\ge1.0$, $R_{\rm in}\le0.4$); X-points in
$R\in[0.25,1.5]$, $|Z|\ge0.4$. The read-outs extract the LCFS with the scorer's extractor and
`mast_envelope.npz` `mask_coarse`.

**$q_{95}$.** $q_{\rm map}=|F_b|W/2\pi$, $F_b=\mu_0 N_{\rm TF} I_{\rm TF}/2\pi$ (vacuum, `fb_from_bt`).
$\mathrm{well}=\mathrm{clip}((q_{\rm map}-8)/(17-8),0,1)$, $\mathrm{div}=1+0.35(1-w)(1-b)$,
$\mathrm{hole}=2b(1-b)^3$, $q_{95}=(0.55\,\mathrm{well}+(1-\mathrm{well})\,\mathrm{div}+\mathrm{hole})\,q_{\rm map}$.
Fallback $q^\*=5a^2|B_t|/(R_0 I_p)$.

**$\beta_N$.** $100\langle p\rangle 2\mu_0 a/(B_t I_p)$, $\langle p\rangle$ over Thomson chords with
$R\in[r_{\rm in},r_{\rm out}]$ of the LCFS; clip $[0.2,4]$, 7-frame median, $\times(1-0.2w)$,
$\times s_\beta$ ($=1$); fallback 1.3.

**Hybrid scalars.** MSE-only form (E1, E2): $q\leftarrow\mathrm{clip}(a_q q+b_q,2,25)$,
$\beta_N\leftarrow\mathrm{clip}(a_b\beta_N+b_b,0.2,5)$ with $a_q=0.626$, $b_q=1.957$, $a_b=1.013$,
$b_b=0.093$ fitted on FAIR-MAST (E5). From E3 on, $q_{95}$ and $\beta_N$ come from the M9 scalar heads.

## M9

* Features (`models/m9/features.py`): 14 currents + 131 $T_e$ + 131 $n_e$ (core) + 16 + 16 (edge,
  zeroed) = 308 per EFIT frame; linear interpolation onto `efit_times`, zero outside the Thomson clock.
* Augmentation (`network.augment_temporal`): per shot, centred 9-frame moving average with clamped
  edges and its central difference over ±4 frames → 924; standardise with stored train mean/std,
  clip ±10.
* Network (`network.CoeffDualContext`): present stem 308→512 + 4 residual blocks; context stem
  616→512 + 3 residual blocks; $h\leftarrow h+g\,P c$ with gate $g$ initialised to 0, $z$ = linear head($h$);
  dropout 0.05; output 60 PCA coefficients; $\psi=\mu+C^\top z$.
* Training (`models/m9/train.py`): split/QC/$|I_p|$-window as in DATA_PROVENANCE §4; PCA K = 60
  (randomized SVD, seed 42, ≤ 200,000 shot-balanced frames); AdamW(1e-3, wd 1e-4), grad clip 1.0,
  50 epochs, best validation epoch kept; init seeds 101/202/303.

## FEC derived quantities (consistency term)

$R_{\rm axis},Z_{\rm axis}$ from the O-point; $\kappa$, $\delta_{\rm top}$, $\delta_{\rm bot}$ from the
LCFS extremes; $V=2\pi\int_A R\,dR\,dZ$; $\ell_i$ in the ITER $\ell_i(2)$ form $\langle B_p^2\rangle_V/(\mu_0I_p/\ell_p)^2$
(`fec_scoring/derive.py`). Both prediction and EFIT pass through the same code.
