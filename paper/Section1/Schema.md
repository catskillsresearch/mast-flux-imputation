# Input vector and clocks

Per shot the model inputs are:

* **Actuators:** 13 current channels on a 0.2 ms (5 kHz) clock spanning $-2000$ to $4000$ ms
  (30,000 samples): `sol`, `p2u`, `p2l`, `p3u`, `p3l`, `p4u`, `p4l`, `p5u`, `p5l`, `p6u`, `p6l`,
  `tf`, `efps` (kA, or kA·turn for the P circuits), plus $I_p$.
* **Kinetic profiles:** core Thomson $T_e, n_e$ on 131 channels at about 2 ms cadence (130 chord
  radii in the FEC rows) and edge Thomson on 16 channels at $R\approx1.33$–$1.49$ m. All chords lie
  on the midplane $Z=0$ [@scannell2010].
* **Geometry:** the 812-row coil table (656 solenoid rows, 156 P2–P6 elements; centre, width,
  height), the 146-chord Thomson table and the $65\times65$ grid
  $R\in[0.06,2.00]$ m, $Z\in[-2,2]$ m.
* **Output clock:** `efit_times` (ms). Inputs are interpolated onto it.

There are no magnetic-probe ($B_\theta$ pick-up) or flux-loop channels. The labels, however, are
EFIT++ reconstructions that *were* constrained by those sensors.
