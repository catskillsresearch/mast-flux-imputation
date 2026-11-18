# Conclusion

Without magnetic sensors, a forward Grad–Shafranov solver driven by measured coil currents, plasma
current and Thomson scattering recovers MAST poloidal flux with $R^2_\psi=0.917210$ on a sealed
archive holdout, but its boundary geometry and scalar read-outs are biased. A PCA-basis neural
surrogate trained on EFIT++ flux of 3,622 QC-kept archive shots (3,265 contributing frames after
the $|I_p|$ window) raises $R^2_\psi$ to 0.997771 and brings the LCFS within
about 5% of the major radius, giving a composite FEC score of 0.862677 in the hybrid configuration.
The surrogate learns from magnetics-based reconstructions and is not physics-informed; the remaining
weakness is in $q_{95}$ and $\beta_N$, which still come from simplified forward read-outs. The
repository, the paper sources and all result tables are reproducible from public data, and the
documentation now states exactly what each component was fitted to.
