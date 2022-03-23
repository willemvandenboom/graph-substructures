""""
This script produces the figure for the simulation study on block structure
recovery in van den Boom et al. (2022, arXiv:2203.11664).
"""

import igraph
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import graph_substructures


def get_rand_index(res, z_true):
    """
    Rand index
    
    Compute the Rand index of the cluster allocation that minimizes Binder's
    loss function.
    """
    z_MCMC = res["Group samples"]
    
    # Find the cluster allocation that minimizes Binder's loss function.
    # Split the Gibbs samples to estimate the transition probabilities and
    # minimize Binder's loss on different posterior samples.
    recorded, p = z_MCMC.shape
    n_train = recorded // 2
    n_test = n_train

    sim_mat_train = np.zeros((p, p))

    for i in range(1, p):
        sim_mat_train[i, :i] = np.mean(
            z_MCMC[::2, i, np.newaxis] == z_MCMC[::2, :i], axis=0
        )

    tmp_mat = np.tril(sim_mat_train - 0.5, k=-1)
    Binder_g = -np.inf

    for i in range(n_test):
        ind = 2*i + 1
        tmp = np.sum(tmp_mat * np.equal.outer(z_MCMC[ind, :], z_MCMC[ind, :]))

        if tmp > Binder_g:
            Binder_g = tmp
            i_Binder = ind

    z_Binder = z_MCMC[i_Binder, :]
    return sklearn.metrics.rand_score(z_true, z_Binder)


def simulation(p, n, K_true):
    """One replicate of the simulation study"""
    z_true = rng.integers(K_true, size=p)
    adj_true = np.equal.outer(z_true, z_true)
    np.fill_diagonal(a=adj_true, val=False)
    edge_prob = 0.2

    for i in range(0, p):
        for j in range(i + 1, p):
            if not adj_true[i, j]:
                adj_true[i, j] = rng.random() < edge_prob
                adj_true[j, i] = adj_true[i, j]

    G_true = igraph.Graph.Adjacency(adj_true.tolist(), mode=1)

    Omega = graph_substructures.wwa.rgwish_identity(
        G_true, graph_substructures.df_0, rng
    )

    data = rng.multivariate_normal(
        mean=np.zeros(p), cov=np.linalg.inv(Omega), size=n
    )

    U = data.T @ data
    U_upper = np.abs(np.triu(m=U, k=1))
    U_median = np.median(U_upper[U_upper.nonzero()])

    G_init = igraph.Graph.Adjacency(
        matrix=(U_upper > U_median).tolist(), mode="undirected"
    )

    res_list = []
    print("Working on SBM...")

    res_list.append(graph_substructures.MCMC_SBM(
        G_init=G_init, data=data, burnin=5 * 10**2, recorded=10**3, rng=rng
    ))
    
    print("Working on SICS...")
    
    res_list.append(graph_substructures.MCMC_SICS(
        G_init=G_init, data=data, burnin=10**3, recorded=5 * 10**3, rng=rng
    ))
    
    print("Working on Sun et al. (2014)...")

    res_list.append(graph_substructures.MCMC_Sun(
        data=data, burnin=5 * 10**2, recorded=10**3, rng=rng
    ))
    
    return [get_rand_index(res, z_true) for res in res_list]


rng = np.random.Generator(np.random.SFC64(seed=0))
n_rep = 50  # Number of replicates

# Run the simulations with K fixed.
n_list = [20, 100, 500, 1000]
n_sim = len(n_list)
sim_res = np.empty((n_sim, n_rep, 3))

for ind in range(n_sim):
    n = n_list[ind]
    
    for r in range(n_rep):
        print("Working on n =", n, "| r =", r)
        sim_res[ind, r, :] = simulation(p=20, n=n, K_true=4)


# Run the simulations with n fixed.
K_list = [2, 3, 4, 5]
n_sim = len(K_list)
sim_res2 = np.empty((n_sim, n_rep, 3))

for ind in range(n_sim):
    K_true = K_list[ind]
    
    for r in range(n_rep):
        print("Working on K_true =", K_true, "| r =", r)
        sim_res2[ind, r, :] = simulation(p=20, n=500, K_true=K_true)


# Plot the results.
n_setup = 3  # Number of algorithms used
CI_width = 0.6 / n_setup
fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

for ind in range(2):
    for setup_ind in range(n_setup):
        Rand_mat = [sim_res, sim_res2][ind]
        x_offset = 0.8 * (setup_ind - 0.5*(n_setup - 1)) / n_setup
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][setup_ind]

        for p_ind in range(n_sim):
            Rand_vec = Rand_mat[p_ind, :, setup_ind]

            # Bootstrapping to get 2.5th and 97.5th percentiles
            Rand_CI = np.percentile(a=rng.choice(
                a=Rand_vec, size=(1000, n_rep), replace=True
            ).mean(axis=1), q=[2.5, 97.5])

            axs[ind].add_patch(plt.Rectangle(
                (p_ind + x_offset - 0.5*CI_width, Rand_CI[0]), CI_width,
                Rand_CI[1] - Rand_CI[0],
                # We do a manual alpha (color transparancy) as otherwise extra
                # rectangle lines appear in the PDF output.
                color=1.0 - 0.3*(1.0 - np.array(matplotlib.colors.to_rgb(col)))
            ))
            
            ls = ["solid", "dashed", "dotted"][setup_ind]

            if p_ind == 0:  # Set label only once
                axs[ind].hlines(
                    y=Rand_vec.mean(), xmin=p_ind + x_offset - 0.5*CI_width,
                    xmax=p_ind + x_offset + 0.5*CI_width, linestyles=ls,
                    color=col, label=[
                        "Stochastic blockmodel", "SICS", "Sun et al. (2014)"
                    ][setup_ind]
                )
            else:
                axs[ind].hlines(
                    y=Rand_vec.mean(), xmin=p_ind + x_offset - 0.5*CI_width,
                    xmax=p_ind + x_offset + 0.5*CI_width, linestyles=ls,
                    color=col
                )
    
    axs[ind].set_xticks(np.arange(n_sim))
    axs[ind].set_xticklabels([n_list, K_list][ind])

    axs[ind].set_xlabel(
        "Number of " + [r"observations $n$", r"clusters $K$"][ind]
    )

    axs[ind].set_ylabel("Rand index")
    axs[ind].set_ylim([0.2, 1.05])

axs[0].legend()
fig.savefig("structure_recovery.pdf", bbox_inches="tight")
