# Provenance: documentation versus code

Reading the source line by line led to the following clarifications and updates of the earlier
documentation. The material ones are:

1. *"No FAIR-MAST EFIT++ / archive reconstructions."* M9's targets and PCA basis are FAIR-MAST
   EFIT++ $\psi$ from 3,622 archive shots; the training code itself notes the pool overlaps the hidden
   test folds.
2. *"`mast_envelope.npz` not used as a mask."* OSF's read-outs extract the LCFS with the scorer's
   envelope mask.
3. *"Zero fitting against the 3 demo shots."* OSF shape choices are justified by demo $R^2$ values,
   and the M9 run's acceptance gate is demo $R^2_\psi\ge0.9976$.
4. *"$c^{*}, s^{*}$ are frozen code hyperparameters."* They were fitted from leaderboard probes on the
   public-test fold; one docstring gives $s^{*}=1.000000$ where the code uses 1.078508.
5. *"Paramagnetic $c=1.20$ baseline"* and *"one-shot"* OSF: neither describes the solver
   (Sections 3.5 and 3.7).
6. *Ingest and demo differ in "recording / $\psi$ scale."* The flux is bit-identical; the frame
   populations differ.
7. *"No magnetic sensors."* True for model inputs; the EFIT++ labels used them.

The full list of seventeen items is kept in `DATA_PROVENANCE.md` of the repository.
