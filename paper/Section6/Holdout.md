# Sealed holdout

Sealed-80 holdout (seed 42), restricted to diverted frames: 65 shots, 3,449 frames.

@import "../generated/holdout_main.md"

Table: Composite score and terms on the sealed holdout.

@import "../generated/run_facts.md"

Table: Run facts. LCFS fail is the fraction of frames whose predicted LCFS could not be extracted;
derive fail the fraction of non-computable consistency scalars.

M9 + OSF improves every term. The flux itself is nearly solved by M9 ($R^2_\psi=0.997771$, pooled;
per-shot mean 0.996645), and the LCFS distance falls from 0.266 to 0.055 of the major radius. OSF,
with no flux labels at all, reaches $R^2_\psi=0.917210$ but its geometry is biased enough that every
consistency $R^2$ is negative. The scalar term is weak for both, because both use the OSF
read-outs.
