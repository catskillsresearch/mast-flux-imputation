# Machine constants versus calibrated parameters

`models/osf/calibration.py` splits the symbols of the source file into four groups. The values are
copied exactly; there is no fitting code.

| group | count | examples | origin |
|---|---|---|---|
| `MACHINE_SPEC` | 12 | $N_{\rm TF}=24$, $\sigma_{\rm ax}=+1$, column names | MAST hardware, FEC schema |
| `GRID_GATES` | 15 | O-point box, X-point box, compactness gates | numerical acceptance on the $65\times65$ grid |
| `SWITCHES` | 12 | measured $I_p$ for read-out, LCFS-window pressure | code path |
| `CALIBRATED` | 38 | $r_\star, I_\star$, ellipse, $k$, $\lambda_c$, $q$ law, $\beta$ law | calibration (Section 5) |

The 38 calibrated values are those appearing in \eqref{eq:clocks}–\eqref{eq:qlaw} and the
$\beta_N$ post-processing:

* clocks: $r_\star=0.34$, $I_\star=16.3$ kA;
* ellipse: 0.82, 0.10, 0.02, 0.009, 0.004, 0.50, 0.08, 0.08, 0.42, 0.18;
* profiles: $\bar\gamma=-1$; $k$ coefficients 0.70, 0.30, 0.15, 0.20; two zero offsets;
* coil multipliers: 1.8, $-1$, 0.105, 1.15, $-0.18$, 0.5, 0.10;
* $q$ law: 17, 8, 0.55, 0.35, 2, 3;
* $\beta$ law: 7-frame median, 0.20, clip $[0.2,4]$, fallback 1.3, $s_\beta=1$.
