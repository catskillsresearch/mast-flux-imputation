# Limitations

* **$\beta_N$ at the end of the shot.** On the validation shots the last five frames carry 51%, 94%
  and 77% of the squared $\beta_N$ error; on shot 28350 the $\beta_N$ RMSE is 0.139 with them and
  0.036 without. Holdout $R^2_{\beta_N}$ is 0.677889, the lowest of the three main terms.
* **Boundary tips and X-points.** 32.4% of holdout frames have an LCFS tip gap of at least 50 mm,
  the 95th-percentile Hausdorff distance is 102.3 mm and the mean X-point offset 34.3 mm. The soft
  read-outs track the scorer's $\delta_{\rm top}$ and $\delta_{\rm bot}$ errors only weakly
  (correlation 0.177 and $-0.056$), so the calibration acts on the boundary tips mainly through the
  tip band.
* **Shape scalars on the validation shots.** $R^2_\kappa$ is 0.827382 and $R^2_{\delta_{\rm bot}}$
  0.427937 on the three shots pooled (0.063399 on 28348), against 0.900994 and 0.761293 on the
  holdout.
* **Internal inductance.** $R^2_{\ell_i}=0.674204$ on the holdout, the lowest consistency term
  there; $\ell_i$ is derived from $|\nabla\psi|$ and depends on the plasma current profile, which
  the inputs constrain only through the Thomson profiles.
* **Labels are magnetics-based.** EFIT++ used probes and flux loops; the task is imputation of a
  magnetics-constrained reconstruction, and its errors are inherited.
* **Small validation set.** The three validation shots are consecutive (115 frames); pooled $R^2$
  values on them are noisy, and $R^2_{Z_{\rm axis}}$ is negative because $Z_{\rm axis}$ barely varies
  within them.
* **Non-causal context.** The model uses EFIT frames up to four steps ahead; it is an offline
  smoother.
