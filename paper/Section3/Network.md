# Network

Each member $f_s$ (`CoeffDualContext`) has two residual encoders [@he2016]:

$$
\begin{aligned}
h &= \mathrm{Res}^{(4)}_{512}\big(\mathrm{ReLU}(W_h\,\tilde x_{[0,308)})\big), &
c &= \mathrm{Res}^{(3)}_{512}\big(\mathrm{ReLU}(W_c\,\tilde x_{[308,924)})\big),\\
h' &= h + g\,P\,c, & z &= W_z\,h' + b_z\in\mathbb R^{60},
\end{aligned}
\label{eq:net}
$$

where $\mathrm{Res}^{(n)}_{512}$ is a stack of $n$ residual blocks of width 512 with dropout 0.05,
$P$ is a $512\times512$ linear projection and $g$ is a scalar gate initialised to 0, so training
starts from the present-frame encoder alone and learns how much temporal context to admit. The
predicted flux is reconstructed with \eqref{eq:pca}.
