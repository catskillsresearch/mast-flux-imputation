# PCA basis of EFIT++ flux

Each EFIT++ map is flattened to $\mathbf y_t\in\mathbb R^{4225}$. With $\mu$ the sample mean and
$C\in\mathbb R^{60\times4225}$ the leading 60 principal directions (orthonormal rows)
[@pearson1901; @hotelling1933], the reduced representation and its reconstruction are

$$
z_t = C(\mathbf y_t-\mu)\in\mathbb R^{60},\qquad \hat{\mathbf y}_t = \mu + C^{\top}z_t .
\label{eq:pca}
$$

This is the proper orthogonal decomposition of the flux ensemble [@berkooz1993]. Because $C$ has
orthonormal rows, MSE on $z$ equals pixel MSE of the *projected* target. The basis is fitted with
randomized SVD [@halko2011] (scikit-learn [@pedregosa2011], seed 42, three power iterations) on at
most 200,000 frames drawn shot-balanced from the training shots, after the $|I_p|$ window of the
next note. The PCA basis and mean are therefore built from FAIR-MAST EFIT++ labels (Section 8.2).
