""""
This script produces the results for the simulation study with the karate club
network in van den Boom et al. (2022, arXiv:2203.11664).

The results include the figure with posterior similarity matrices as well as
the figure for the comparison of the proposed Bayes factor computation with the
harmonic mean approach.
"""

import igraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.integrate

import graph_substructures


# Gamma prior on the Dirichlet concentration parameter alpha
graph_substructures.a_alpha = 5.0
graph_substructures.b_alpha = graph_substructures.a_alpha

# Gamma prior on the Dirichlet concentration parameter nu
graph_substructures.a_nu = graph_substructures.a_alpha
graph_substructures.b_nu = graph_substructures.a_nu

rng = np.random.Generator(np.random.SFC64(seed=0))

G_true = igraph.Graph.Adjacency(
    (nx.to_numpy_matrix(nx.karate_club_graph()) > 0).tolist(), mode=1
)

p = G_true.vcount()
df_0 = 3.0 # Degrees of freedom of the Wishart prior
K_true = graph_substructures.wwa.rgwish_identity(G_true, df_0, rng)
res_list = []
n_list = [10**4, 10**3, 10**2, 10]

for n in n_list:
    print("\nWorking on n =", n)

    data = rng.multivariate_normal(
        mean=np.zeros(p), cov=np.linalg.inv(K_true), size=n
    )
    
    res_list.append(graph_substructures.MCMC_SBM(
        G_init=G_true, data=data, burnin=10**3, recorded=5 * 10**3, rng=rng
    ))


# Plot the results.
# Order from Tan & De Iorio (2019, doi:10.1177/1471082X18770760)
order = np.array([
    1, 5, 7, 11, 6, 4, 17, 8, 13, 2, 18, 22, 14, 20, 12, 3, 10, 29, 25, 34, 24,
    30, 26, 33, 32, 27, 28, 15, 31, 9, 16, 21, 23, 19
]) - 1

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 11))

for n_ind in range(len(n_list)):
    z_MCMC = res_list[n_ind]["Group samples"]

    # Compute posterior similarity matrix.
    # It contains the posterior probabilities that two nodes belong to the same
    # group.
    sim_mat = np.eye(p)

    for i in range(1, p):
        sim_mat[i, :i] = np.mean(
            z_MCMC[:, i, np.newaxis] == z_MCMC[:, :i], axis=0
        )

    sim_mat = sim_mat + sim_mat.T
    i = n_ind // 2
    j = n_ind % 2

    im = axs[i, j].pcolormesh(
        np.rot90(sim_mat[order][:, order]), cmap="Greys", vmin=0.0, vmax=1.0
    )

    axs[i, j].axis("image")
    labels = (order + 1).astype(str)
    labels[0] = "Mr Hi"
    labels[19] = "John A."
    axs[i, j].set_xticks(np.arange(p) + 0.5)
    axs[i, j].set_xticklabels(labels)

    axs[i, j].tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False
    )

    plt.setp(axs[i, j].get_xticklabels(), rotation=-90)
    axs[i, j].set_yticks(np.arange(p)+0.5)
    axs[i, j].set_yticklabels(np.flip(labels))
    axs[i, j].tick_params(labelsize=7)
    
    axs[i, j].set_title(
        [r"$n = 10^4$", r"$n = 10^3$", r"$n = 10^2$", r"$n = 10$"][n_ind]
    )

cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.015])
bar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
bar.set_label(r"Posterior similarity $\mathrm{Pr}(z_i = z_j \mid Y)$")
fig.savefig("karate.pdf", bbox_inches="tight")


# Compute the Bayes factors.
print("Prior probability of only one cluster:")
prior_prob = scipy.integrate.quad(
    lambda nu: np.exp(
        scipy.special.loggamma(nu + 1.0) - scipy.special.loggamma(nu + p) \
            + scipy.special.loggamma(p) \
            + graph_substructures.a_nu*np.log(graph_substructures.b_nu) \
            - scipy.special.loggamma(graph_substructures.a_nu) \
            + (graph_substructures.a_nu - 1.0)*np.log(nu) \
            - graph_substructures.b_nu*nu
    ),
    0, np.inf
)[0]

print(prior_prob)

for n_ind in range(len(n_list)):
    print("Bayes factor for n =", n_list[n_ind])

    print(
        np.mean(res_list[n_ind]["Group samples"].max(axis=1) == 0) / prior_prob
    )


# Plot for the proposed Bayes factor computation vs harmonic mean approach
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

for n_ind in [2, 3]:
    tmp = res_list[n_ind]["Group samples"].max(axis=1) == 0
    log_prob = np.log(np.cumsum(tmp)) - np.log(np.arange(1, 1 + len(tmp)))
    log_prob[log_prob == -np.inf] = -1e6

    tmp = res_list[n_ind]["log_lik_MCMC"]
    tmp_min = tmp.min()

    log_lik = np.log(
        np.cumsum(np.exp(tmp_min - tmp))
    ) - tmp_min - np.log(np.arange(1, 1 + len(tmp)))
    
    ax1 = axes[n_ind - 2]
    ax2 = ax1.twinx()
    ax1.set_zorder(1)
    ax1.set_frame_on(False)
    offset = log_prob[-1] - log_lik[-1]
    ax1.plot(log_prob, zorder=10)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    ax2.plot(log_lik + offset, linestyle="dashed", color=cols[1])
    
    ax1.set_xlabel("MCMC iteration")

    ax1.set_ylabel(
        r"MCMC estimate of $\log \{ p(z^*\mid Y) \}$", color=cols[0]
    )

    ax2.set_ylabel(
        r"Harmonic mean estimate of $-\log \{ p(Y \mid \mathcal{M}) \}$",
        color=cols[1]
    )
    
    ax1.set_ylim((-7.5, -2.5))
    offset_round = int(offset.round())
    y_ticks = np.unique(ax1.get_yticks().astype(int)) + offset_round
    ax2.set_yticks(y_ticks - offset_round)
    ax2.set_yticklabels(y_ticks)
    ax2.set_ylim((-7.5, -2.5))
    
    ax1.set_title(
        [r"$n = 10^4$", r"$n = 10^3$", r"$n = 10^2$", r"$n = 10$"][n_ind]
    )

fig.tight_layout()
fig.savefig("karate_BF.pdf", bbox_inches="tight")
