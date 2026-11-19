# Input vector and clocks

Per shot the model reads:

* **Actuators:** 13 current channels on a 0.2 ms (5 kHz) clock spanning $-2000$ to $4000$ ms
  (30,000 samples): `sol`, `p2u`, `p2l`, `p3u`, `p3l`, `p4u`, `p4l`, `p5u`, `p5l`, `p6u`, `p6l`,
  `tf`, `efps` (kA, or kA·turn for the P circuits), plus $I_p$.
* **Kinetic profiles:** core Thomson $T_e, n_e$ on 131 channels at about 2 ms cadence, on the
  midplane $Z=0$ [@scannell2010]. Edge Thomson (16 channels at $R\approx1.33$–$1.49$ m) is present
  in the rows but zeroed by the model.
* **Output clock:** `efit_times` (ms). Inputs are interpolated onto it.

There are no magnetic-probe ($B_\theta$ pick-up) or flux-loop channels. The labels, however, are
EFIT++ reconstructions that *were* constrained by those sensors.
