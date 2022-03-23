""""
This script produces the results for the application to mutual fund data in
van den Boom et al. (2022, arXiv:2203.11664).
"""

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special

import graph_substructures


rng = np.random.Generator(np.random.SFC64(seed=0))
data = np.loadtxt(fname="ExampleSection6.txt", dtype=float, delimiter=" ")
n, p = data.shape

# Quantile-normalize the data to marginally follow a standard Gaussian
# distribution.
for j in range(p):
    data[:, j] = scipy.special.ndtri(
        pd.Series(data[:, j]).rank(method="average") / (n + 1)
    )


U = data.T @ data
U_upper = np.abs(np.triu(m=U, k=1))
U_median = np.median(U_upper[U_upper.nonzero()])

G_init = igraph.Graph.Adjacency(
    matrix=(U_upper > U_median).tolist(), mode="undirected"
)

# Run the MCMC for the degree-corrected stochastic blockmodel.
res_SBM = graph_substructures.MCMC_SBM(
    G_init=G_init, data=data, burnin=5 * 10**3, recorded=10**4, rng=rng,
    init_K1_M1=True
)

# Run the MCMC for the model with the SICS prior.
res_SICS = graph_substructures.MCMC_SICS(
    G_init, data, burnin=10**4, recorded=10**5, rng=rng, diag=True
)


# Plot the results.
mods = [13, 43, 50]  # Module cutoffs per fund type
fig, axs = plt.subplots(ncols=2, figsize=(10, 6))

for ind in range(2):
    z_MCMC = [res_SBM, res_SICS][ind]["Group samples"]
    p = len(z_MCMC[0, :])

    # Compute posterior similarity matrix.
    # It contains the posterior probabilities that two nodes belong to the same
    # group.
    sim_mat = np.eye(p)

    for i in range(1, p):
        sim_mat[i, :i] = np.mean(
            z_MCMC[:, i, np.newaxis] == z_MCMC[:, :i], axis=0
        )

    sim_mat = sim_mat + sim_mat.T

    im = axs[ind].pcolormesh(sim_mat, cmap="Greys", vmin=0.0, vmax=1.0)
    axs[ind].axis("image")
    axs[ind].invert_yaxis()

    for i in mods:
        axs[ind].plot(
            [0, p], [i, i], color="red", linestyle=(0, (0.1, 2)),
            dash_capstyle="round"
        )
        
        axs[ind].plot(
            [i, i], [0, p], color="red", linestyle=(0, (0.1, 2)),
            dash_capstyle="round"
        )
    
    axs[ind].xaxis.set_label_position("top") 
    axs[ind].set_xlabel(r"Node index $j$")
    axs[ind].set_ylabel(r"Node index $i$")

    axs[ind].set_title(
        ["Stochastic blockmodel", "Southern Italian community structure"][ind]
    )
    
    axs[ind].tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False
    )

    labels = np.arange(start=5, stop=p + 1, step=5, dtype=int)
    axs[ind].set_xticks(labels - 0.5)
    axs[ind].set_yticks(labels - 0.5)
    axs[ind].set_xticklabels(labels)
    axs[ind].set_yticklabels(labels)

bar = fig.colorbar(im, ax=axs, shrink=0.5, orientation="horizontal", pad=0.05)
bar.set_label(r"Posterior similarity $\mathrm{Pr}(z_i = z_j\mid Y)$")
fig.savefig("fund.pdf", bbox_inches="tight")
