# Score

The score box of the viewer is given for each validation shot scored alone, for the three
validation shots pooled, and for the sealed holdout (65 shots, 3,449 frames).

@import "../generated/model_scorebox.md"

Table: Composite score $S$, its terms and the seven consistency $R^2$ values. Each column is one fold; $R^2$ values are pooled over the frames of the fold.

On the sealed holdout the model reaches $S=0.934444$. The flux term is $R^2_\psi=0.997832$; the scalar
terms are $R^2_{q_{95}}=0.913296$ and $R^2_{\beta_N}=0.677889$; $1-D_{\rm LCFS}=0.938632$ and the
Consistency term is 0.862171, with every consistency $R^2$ between 0.674204 ($\ell_i$) and 0.961958
($Z_{\rm axis}$).

On the three validation shots pooled $S=0.899573$, with $R^2_\psi=0.998879$ and
$R^2_{q_{95}}=0.976192$, both above their holdout values. The difference in $S$ comes from the
Consistency term (0.659619): $Z_{\rm axis}$ varies by only a few millimetres within these shots, so a
millimetre-scale offset gives $R^2_{Z_{\rm axis}}=-5.467191$, which is clipped to zero, and
$\delta_{\rm bot}$ reaches $R^2=0.427937$. Shot 28348 has the lowest score, $S=0.874374$, with
$R^2_{\beta_N}=0.533545$ and $R^2_{\delta_{\rm bot}}=0.063399$; shots 28350 and 28351 score 0.910541
and 0.905806.
