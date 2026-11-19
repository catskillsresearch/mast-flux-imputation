# Sealed holdout

After the pick, the holdout (65 shots, 3,449 diverted frames) was scored once for E1, E4 and OSF
with the E5 maps, with the same scorer and audit as the shipped models.

@import "../../generated/exp_holdout.md"

Table: Sealed holdout. $\mathcal R$ relative to the shipped hybrid.

E4, trained and calibrated from FAIR-MAST alone, reaches $S=0.931421$ against $0.862677$ for the
shipped hybrid and $0.874196$ for E1. $R^2_\psi$ is the same for all three to four decimals
(0.997837, 0.997771, 0.997808); the gain is in the scalars: $R^2_{q_{95}}$ rises from 0.342298
(shipped) and 0.444019 (E1) to 0.916470, and $R^2_{\beta_N}$ from 0.394260 and 0.391840 to
0.689018. Of the 30 audit metrics, E4 is at least as good as the shipped hybrid on 24; it is worse
on $R^2(Z_{\rm axis})$, $R^2(\delta_{\rm top})$, the O- and X-point offsets, the X-point-lost rate
and the topology-flip rate. Against E1 it is better on the tip-gap rate (23.1% against 24.9%), the
p95 Hausdorff distance (97.9 against 104.2 mm) and the mean Hausdorff distance, and worse on the
O- and X-point offsets and the flip rate (3.0% against 2.6%), so its composite $\mathcal R$ is
0.9906 against 0.9742 for E1.

OSF with the E5 maps reaches $S=0.640558$ against $0.618634$ with $c^*$, $s^*$: the
FAIR-MAST-only read-out generalises better than the probe-fitted one.

Between the validation shots and the holdout, $S$ falls for the shipped hybrid (0.871921 to
0.862677) and rises for E1 (0.863300 to 0.874196) and E4 (0.867116 to 0.931421).
