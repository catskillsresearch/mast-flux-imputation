# Sealed holdout

Sealed-80 holdout (seed 42), restricted to diverted frames: 65 shots, 3,449 frames, scored once per
model.

@import "../generated/res_holdout_main.md"

Table: Composite score, its terms and route quantities on the sealed holdout. OSF uses the E5 read-out maps; E1 is the hybrid trained with coefficient MSE only. Tip gap: share of frames with an LCFS tip gap of at least 50 mm; X-pt: mean X-point offset; flips: limited/diverted flips.

E6 has the highest score, $S=0.934444$, against 0.931421 for E4 and 0.874196 for the MSE-only
hybrid E1. The flux term is the same for the three trained models to four decimals
($R^2_\psi=0.997832$ for E6). The scalar terms come from the trained heads ($R^2_{q_{95}}=0.913296$,
$R^2_{\beta_N}=0.677889$), and the Consistency term rises to 0.862171 from 0.841297 (E1) and
0.838024 (E4). The boundary route moves the other way: E6 has more frames with a tip gap of at
least 50 mm (32.4% against 24.9% for E1 and 23.1% for E4), a larger mean X-point offset (34.3 mm
against 20.5 and 22.0 mm), more flips (3.6%) and a slightly lower $1-D_{\rm LCFS}$ (0.938632
against 0.944528 and 0.945939). The curve-distance terms pull the shape scalars towards EFIT++
along each shot, while the LCFS and X-point position carry only a small weight in the loss.

OSF alone, with no flux labels, reaches $R^2_\psi=0.917210$, but its geometry is biased enough that
every consistency $R^2$ is negative.
