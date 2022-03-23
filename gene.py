""""
This script produces the results for the application to gene expression data in
van den Boom et al. (2022, arXiv:2203.11664).
"""

import time

import igraph
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special

import wwa


np.seterr(divide="ignore")  # Enable np.log(0.0).
rng = np.random.Generator(np.random.SFC64(seed=1))

# Read in the data as extracted from The Cancer Genome Atlas using the R script
# `gene.R`.
csv_file = np.loadtxt(fname="gene.csv", dtype=str, delimiter=",")

cancer_type = csv_file[1:, 0]
gene_names = csv_file[0, 1:]
data = csv_file[1:, 1:].astype(float)
p = data.shape[1]    

cancer_type_unique = np.unique(cancer_type)
q = len(cancer_type_unique)
n_x = np.empty(q, dtype=int)
U_x = np.empty((q, p, p))

for x in range(q):
    data_x = data[cancer_type == cancer_type_unique[x], :]
    n_x[x] = data_x.shape[0]
    
    # Quantile-normalize the data to marginally follow a standard Gaussian
    # distribution.
    for j in range(p):
        data_x[:, j] = scipy.special.ndtri(
            pd.Series(data_x[:, j]).rank(method="average") / (n_x[x] + 1)
        )
    
    U_x[x] = data_x.T @ data_x

df_0 = 3.0 # Degrees of freedom of the Wishart prior
rate_0 = np.eye(p) # Rate matrix of the Wishart prior
s_theta = 1.0  # Prior standard deviation for theta
s_beta = 1.0  # Prior standard deviation for beta
a_alpha = 2.0  # Gamma prior on the Dirichlet concentration parameter alpha
b_alpha = a_alpha
a_nu = a_alpha  # Gamma prior on the Dirichlet concentration parameter nu
b_nu = a_nu
gamma = 0.5  # Prior probability of z_x[x, i] == z_prime[i]

df_x = df_0 + n_x
rate_x = rate_0 + U_x


def MCMC_update(
    G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu, rng
):
    """MCMC step"""
    mu_x = np.tile(np.add.outer(theta, theta), (q, 1, 1))
    
    for x in range(q):
        for i in range(p):
            for j in range(i + 1, p):
                if z_x[x, i] == z_x[x, j]:
                    mu_x[x, i, j] += beta_x[x, i]
    
    prob_mu_x = scipy.special.ndtr(mu_x)
    log_lik_x = np.empty(q)
    
    for x in range(q):
        # Update G
        G_x[x], log_lik_x[x] = wwa.MCMC_update(
            G_x[x], prob_mu_x[x], df_x[x], rate_x[x], rng, get_log_lik=True
        )

    _, c_index, c_size = np.unique(c, return_index=True, return_counts=True)

    _, z_index = np.unique(
        np.concatenate((z_prime.flatten(), z_x.flatten())), return_index=True
    )

    S_size = np.bincount(np.concatenate((z_prime.flatten(), z_x[~g_x])))
    theta_star = theta[c_index]

    beta_star = np.concatenate(
        (beta_prime.flatten(), beta_x.flatten())
    )[z_index]
    
    # Albert & Chib (1993) latent variable update of theta and beta
    M = len(c_index)
    K = len(z_index)
    zeta_sum_k = np.zeros((K,))
    zeta_sum_tot = 0.0
    zeta_mat = np.zeros((p, p))
    neighbors_x = q * [None]
    
    for x in range(q):
        tmp = G_x[x].neighborhood(mindist=1)
        neighbors_x[x] = [np.array(tmp[i], dtype=int) for i in range(p)]

        for i in range(p - 1):
            mask = np.zeros(p - i - 1, dtype=bool)
            mask[neighbors_x[x][i][neighbors_x[x][i] > i] - i - 1] = True
            
            zeta = mu_x[x, i, i + 1:] + scipy.special.ndtri(
                mask + (~mask + (2*mask - 1)*prob_mu_x[x, i, i + 1:])*(
                    rng.random(p - i - 1) - mask
                )
            )

            zeta_sum_tot += zeta.sum()
            k = z_x[x, i]
            zeta_sum_k[k] += zeta[z_x[x, i + 1:] == k].sum()
            zeta_mat[i, i + 1:] += zeta
    

    # Update theta_star.
    c_pairs = q * c_size * (c_size - 1) // 2  # Number of pairs in all graphs.
    theta_var = 1.0 / (s_theta**-2 + 4*c_pairs + q*c_size*(p - c_size))
    
    for m in range(M):        
        theta_mean = 0.0
        
        for i in range(p - 1):
            if c[i] == m:
                for j in range(i + 1, p):
                    if c[j] == m:
                        theta_mean += 2.0 * zeta_mat[i, j]
                        
                        for x in range(q):
                            if z_x[x, i] == z_x[x, j]:
                                theta_mean -= 2.0 * beta_x[x, i]
                    else:
                        theta_mean += zeta_mat[i, j] - q*theta[j]
                        
                        for x in range(q):
                            if z_x[x, i] == z_x[x, j]:
                                theta_mean -= beta_x[x, i]
            else:
                for j in range(i + 1, p):
                    if c[j] == m:
                        theta_mean += zeta_mat[i, j] - q*theta[i]
                        
                        for x in range(q):
                            if z_x[x, i] == z_x[x, j]:
                                theta_mean -= beta_x[x, i]
        
        theta_star[m] = rng.normal(
            theta_var[m] * theta_mean, scale=np.sqrt(theta_var[m])
        )
        
        theta[c == m] = theta_star[m]
    
    # Update beta_star.
    S_size_x = np.apply_along_axis(
        func1d=np.bincount, axis=1, arr=z_x, minlength=K
    )
    
    S_n_pairs_k = np.sum(S_size_x * (S_size_x - 1) // 2, axis=0)
    beta_var = 1.0 / (s_beta**-2 + S_n_pairs_k)
    theta_sum = np.zeros(K)
    
    for x in range(q):
        for i in range(p - 1):
            for j in range(i + 1, p):
                if z_x[x, i] == z_x[x, j]:
                    theta_sum[z_x[x, i]] += theta[i] + theta[j]
    
    beta_star = rng.normal(
        loc=beta_var * (zeta_sum_k - theta_sum), scale=np.sqrt(beta_var)
    ).T

    # Update c.
    # This update follows Step 5 of Algorithm 1 of
    # Tan & De Iorio (2019, doi:10.1177/1471082X18770760). That is,
    # conditionally on zeta generated according to the Albert & Chib (1993)
    # latent variable scheme for the probit, we perform Algorithm 2 of
    # Neal (2000, http://www.jstor.org/stable/1390653). An alternative would be
    # Algorithm 8 of Neal (2000) not conditional on zeta.
    zeta_mat += zeta_mat.T
    
    for i in range(p):
        m_cur = c[i]
        c[i] = -1
        c_size[m_cur] -= 1
        log_prob = np.log(np.append(c_size, alpha))  # Dirichlet process
        tmp = zeta_mat[i, :].sum() - q*(theta.sum() - theta[i])

        for x in range(q):
            tmp -= (S_size_x[x, z_x[x, i]] - 1) * beta_star[z_x[x, i]]

        log_prob[:M] += theta_star*tmp - 0.5*q*(p - 1)*theta_star**2
        s_c = 1.0 / np.sqrt(q*(p - 1) + s_theta**-2)
        m_c = s_c**2 * tmp
        log_prob[-1] += np.log(s_c) - np.log(s_theta) + 0.5*(m_c / s_c)**2
        log_prob -= log_prob.max()
        prob = np.exp(log_prob)
        c[i] = rng.choice(a=M + 1, p=prob / prob.sum())
        
        if c[i] == M:
            M += 1
            theta_star = np.append(theta_star, rng.normal(loc=m_c, scale=s_c))
            c_size = np.append(c_size, 1)
        else:
            c_size[c[i]] += 1
        
        # Remove empty clusters, if any.
        if c_size[m_cur] == 0:
            M -= 1
            c[c > m_cur] -= 1
            theta_star = np.delete(theta_star, m_cur)
            c_size = np.delete(c_size, m_cur)
        
        theta[i] = theta_star[c[i]]

    # Update z.
    # This is similar to Algorithm 2 of
    # Neal (2000, http://www.jstor.org/stable/1390653) and does not condition
    # on zeta. A difference from the Algorithm 2 is that the likelihood for the
    # node being considered also depends on the other nodes.
    # Note that Algorithm 8 of Neal (2000) coincides with its Algorithm 2 for
    # our model since a cluster with only one node does not have a likelihood
    # involving beta.

    # Update z_prime.
    for i in range(p):
        mu_ccz = np.add.outer(
            np.add.outer(theta_star, theta_star), np.append(beta_star, 0.0)
        )

        log_edge_prob_ccz = [
            scipy.special.log_ndtr(-mu_ccz), scipy.special.log_ndtr(mu_ccz)
        ]

        g_ind = g_x[:, i].nonzero()[0]
        n_g = len(g_ind)
        k_cur = z_prime[i]
        z_prime[i] = -1
        z_x[g_ind, i] = -1
        S_size_x[g_ind, k_cur] -= 1
        S_size[k_cur] -= 1
        log_prob = np.log(np.append(S_size, nu))
        
        for x in g_ind:
            nb_ind = 0

            for j in range(p):
                if j == i:
                    continue
                
                if nb_ind < len(neighbors_x[x][i]):
                    connected = int(j == neighbors_x[x][i][nb_ind])
                else:
                    connected = 0
                
                log_prob[z_x[x, j]] \
                    += log_edge_prob_ccz[connected][c[i], c[j], z_x[x, j]]

                log_prob[np.arange(K + 1) != z_x[x, j]] \
                    += log_edge_prob_ccz[connected][c[i], c[j], K]

                nb_ind += connected

        log_prob -= log_prob.max()
        prob = np.exp(log_prob)
        z_prime[i] = rng.choice(a=K + 1, p=prob / prob.sum())
        z_x[g_ind, i] = z_prime[i]

        if z_prime[i] == K:
            K += 1
            beta_star = np.append(beta_star, rng.normal(scale=s_beta))
            S_size_x = np.append(S_size_x, np.zeros((q, 1)), axis=1)
            S_size_x[g_ind, -1] = 1
            S_size = np.append(S_size, 1)
        else:
            S_size_x[g_ind, z_prime[i]] += 1
            S_size[z_prime[i]] += 1

        # Remove empty clusters, if any.
        if S_size[k_cur] == 0:
            K -= 1
            z_prime[z_prime > k_cur] -= 1
            z_x[z_x > k_cur] -= 1
            beta_star = np.delete(beta_star, k_cur)
            S_size_x = np.delete(S_size_x, k_cur, axis=1)
            S_size = np.delete(S_size, k_cur)

    # Update (g_x, z_x).
    for i in range(p):
        for x in range(1, q):
            # Sample g_x from conditional which does not condition on z_x.
            k_cur = z_x[x, i]
            z_x[x, i] = -1
            S_size_x[x, k_cur] -= 1
            
            if not g_x[x, i]:
                S_size[k_cur] -= 1
            
            # Remove empty clusters, if any.
            if S_size[k_cur] == 0:
                K -= 1
                z_prime[z_prime > k_cur] -= 1
                z_x[z_x > k_cur] -= 1
                beta_star = np.delete(beta_star, k_cur)
                S_size_x = np.delete(S_size_x, k_cur, axis=1)
                S_size = np.delete(S_size, k_cur)

            mu_ccz = np.add.outer(
                np.add.outer(theta_star, theta_star), np.append(beta_star, 0.0)
            )

            log_edge_prob_ccz = [
                scipy.special.log_ndtr(-mu_ccz), scipy.special.log_ndtr(mu_ccz)
            ]

            log_prob = np.zeros(K + 1)
            nb_ind = 0

            for j in range(p):
                if j == i:
                    continue
                
                if nb_ind < len(neighbors_x[x][i]):
                    connected = int(j == neighbors_x[x][i][nb_ind])
                else:
                    connected = 0
                
                log_prob[z_x[x, j]] \
                    += log_edge_prob_ccz[connected][c[i], c[j], z_x[x, j]]

                log_prob[np.arange(K + 1) != z_x[x, j]] \
                    += log_edge_prob_ccz[connected][c[i], c[j], K]

                nb_ind += connected
            
            log_prob_prior = np.log(np.append(S_size, nu))
            log_prob_prior -= scipy.special.logsumexp(log_prob_prior)
            
            # Log odds of g_x = 1
            log_odds = np.log(gamma) - np.log(1 - gamma) \
                + log_prob[z_prime[i]] \
                - scipy.special.logsumexp(log_prob_prior + log_prob)
            
            g_x[x, i] = rng.random() < 1.0 / (1.0 + np.exp(-log_odds))           
            
            # Udpate z_x[x, i]
            if g_x[x, i]:
                z_x[x, i] = z_prime[i]
                S_size_x[x, z_x[x, i]] += 1
            else:  # g_x[x, i] == False
                # Sample z_x[x, i]
                log_prob += log_prob_prior
                log_prob -= log_prob.max()
                prob = np.exp(log_prob)
                z_x[x, i] = rng.choice(a=K + 1, p=prob / prob.sum())
                
                if z_x[x, i] == K:
                    K += 1
                    beta_star = np.append(beta_star, rng.normal(scale=s_beta))
                    S_size_x = np.append(S_size_x, np.zeros((q, 1)), axis=1)
                    S_size_x[x, -1] = 1
                    S_size = np.append(S_size, 1)
                    mu_K = theta + beta_star[-1]
                else:
                    S_size_x[x, z_x[x, i]] += 1
                    S_size[z_x[x, i]] += 1
    
    # Update alpha per Escobar & West (1995).
    log_gamma = np.log(rng.beta(alpha + 1.0, p))
    r = (a_alpha + M - 1.0) / p / (b_alpha - log_gamma)
    
    alpha = rng.gamma(
        shape=(
            a_alpha + M if rng.random() < r / (r + 1.0) else a_alpha + M - 1.0
        ),
        scale=1.0 / (b_alpha - log_gamma)
    )
    
    # Update nu per Escobar & West (1995).
    p_prime = p + np.sum(~g_x)
    log_gamma = np.log(rng.beta(nu + 1.0, p_prime))
    r = (a_nu + K - 1.0) / p_prime / (b_nu - log_gamma)
    
    nu = rng.gamma(
        shape=a_nu + K if rng.random() < r / (r + 1.0) else a_nu + K - 1.0,
        scale=1.0 / (b_nu - log_gamma)
    )
    
    beta_prime = beta_star[z_prime]

    for x in range(q):
        beta_x[x] = beta_star[z_x[x]]

    return (
        G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu,
        log_lik_x
    )


def MCMC(burnin=10**3, recorded=10**4, verbose=True, plot=False):
    # Initialize.
    G_x = G_x_init.copy()
    c = np.zeros(p, dtype=int)
    theta = -1.4 / 2.0 * np.ones(p)

    z_prime = z_x_init[0, :].copy()
    z_x = z_x_init.copy()
    beta_prime = np.zeros((p,))
    beta_x = np.tile(beta_prime, (q, 1))
    g_x = z_x == z_prime
    alpha = 1.0  # Concentration parameter for the Dirichlet
    nu = 1.0
    
    res_x = np.zeros((q, p, p), dtype=int)
    theta_MCMC = np.empty((recorded, p))
    beta_x_MCMC = np.empty((recorded, q, p))
    g_x_MCMC = np.empty((recorded, q, p), dtype=bool)
    c_MCMC = np.empty((recorded, p), dtype=int)
    z_x_MCMC = np.empty((recorded, q, p), dtype=int)
    z_prime_MCMC = np.empty((recorded, p), dtype=int)
    alpha_MCMC = np.empty(recorded)
    nu_MCMC = np.empty(recorded)
    n_e_x = np.empty((burnin + recorded, q), dtype=int)
    log_lik_x_MCMC = np.empty((recorded, q))
    
    if plot:
        t0 = time.time()
        fig, ax = plt.subplots()
        x_list = np.arange(burnin + recorded)
        line = matplotlib.lines.Line2D([], [])
        ax.add_line(line)
        ylims = [G_x_init[0].ecount() - 1, G_x_init[0].ecount() + 1]
        
        def update_plot(s, ylims, t0):
            t1 = time.time()
            
            if t1 - t0 < 1.0 and s < burnin + recorded - 1:
                return t0
            
            line.set_data(x_list[:(s + 1)], n_e_x[:(s + 1), 0])
            ylims[0] = min(ylims[0], n_e_x[s, 0] - 1)
            ylims[1] = max(ylims[1], n_e_x[s, 0] + 1)
            
            if s > 0:
                ax.set_xlim(0, s)
            
            ax.set_ylim(ylims[0], ylims[1])
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            return t1
    
    for s in range(burnin):
        if verbose: # and s % 100 == 0:
            print("Burnin iteration", s, end="\r")
        
        (
            G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu, _
        ) = MCMC_update(
            G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu,
            rng
        )
        
        for x in range(q):
            n_e_x[s, x] = G_x[x].ecount()
        
        if plot:
            t0 = update_plot(s, ylims, t0)
    
    for s in range(recorded):
        if verbose: # and s % 100 == 0:
            print("Iteration", s, end="\r")
        
        (
            G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu,
            log_lik_x
        ) = MCMC_update(
            G_x, theta, beta_prime, beta_x, c, z_prime, z_x, g_x, alpha, nu,
            rng
        )
        
        for x in range(q):
            res_x[x] += np.array(G_x[x].get_adjacency(0).data)
            n_e_x[burnin + s, x] = G_x[x].ecount()
        
        theta_MCMC[s, :] = theta
        beta_x_MCMC[s] = beta_x
        g_x_MCMC[s] = g_x
        c_MCMC[s, :] = c
        z_x_MCMC[s] = z_x
        z_prime_MCMC[s] = z_prime
        alpha_MCMC[s] = alpha
        nu_MCMC[s] = nu
        log_lik_x_MCMC[s, :] = log_lik_x
        
        if plot:
            t0 = update_plot(burnin + s, ylims, t0)
    
    return {
        "Edge probability": res_x / recorded,
        "Group samples": z_x_MCMC,
        "c_MCMC": c_MCMC,
        "z_prime samples": z_prime_MCMC,
        "Theta samples": theta_MCMC,
        "beta samples": beta_x_MCMC,
        "g_x samples": g_x_MCMC,
        "Alpha samples": alpha_MCMC,
        "Nu samples": nu_MCMC,
        "n_e_x": n_e_x,
        "log_lik_x_MCMC": log_lik_x_MCMC,
        "last G_x": G_x
    }


U_upper = np.abs(np.triu(m=U_x[0], k=1))
U_median = np.median(U_upper[U_upper.nonzero()])

G_init = igraph.Graph.Adjacency(
    matrix=(U_upper > U_median).tolist(), mode="undirected"
)

z_init = np.empty(p, dtype=int)
G_copy = G_init.copy()

# Record vertex IDs as igraph relabels vertices when deleting vertices from the
# graph.
G_copy.vs["vid"] = range(p)

K = 0

while G_copy.vcount() > 0:
    largest_cliq = G_copy.largest_cliques()[0]
    
    for v in largest_cliq:
        z_init[G_copy.vs["vid"][v]] = K
    
    G_copy.delete_vertices(largest_cliq)
    K += 1

print("Initialization has K =", K, "blocks.")
z_init = np.zeros(p, dtype=int)

G_x_init = q * [G_init]
z_x_init = np.tile(z_init, (q, 1))
z_prime_init = z_x_init[0].copy()
res = MCMC(burnin=5 * 10**3, recorded=5 * 10**4)


# Plot the results.
mods = [7, 32, 38]  # Module cutoffs
fig, axs = plt.subplots(ncols=3, figsize=(13, 6))

for ind in range(3):
    cancer_inds = [(0, 0), (0, 1), (1, 1)][ind]
    z_MCMC_both = [res["Group samples"][:, x, :] for x in cancer_inds]

    # Compute posterior similarity matrix.
    # It contains the posterior probabilities that two nodes belong to the same
    # group.
    sim_mat = np.zeros((p, p))

    for i in range(0, p):
        sim_mat[i, :] = np.mean(
            z_MCMC_both[1][:, i, np.newaxis] == z_MCMC_both[0], axis=0
        )

    im = axs[ind].pcolormesh(sim_mat, cmap="Greys", vmin=0.0, vmax=1.0)
    axs[ind].axis("image")

    axs[ind].tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False
    )

    axs[ind].invert_yaxis()
    axs[ind].xaxis.set_label_position("top")

    axs[ind].set_xlabel([
        r"Node index $j$ (breast cancer)", r"Node index $j$ (ovarian cancer)"
    ][cancer_inds[0]])

    axs[ind].set_ylabel([
        r"Node index $i$ (breast cancer)", r"Node index $i$ (ovarian cancer)"
    ][cancer_inds[1]])
    
    for i in mods:
        axs[ind].plot(
            [0, p], [i, i], color="red", linestyle=(0, (0.1, 2)),
            dash_capstyle="round"
        )

        axs[ind].plot(
            [i, i], [0, p], color="red", linestyle=(0, (0.1, 2)),
            dash_capstyle="round"
        )
    
    labels = np.arange(start = 5, stop=p + 1, step=5, dtype=int)
    axs[ind].set_xticks(labels - 0.5)
    axs[ind].set_yticks(labels - 0.5)
    axs[ind].set_xticklabels(labels)
    axs[ind].set_yticklabels(labels)

bar = fig.colorbar(im, ax=axs, shrink=0.3, orientation="horizontal", pad=0.05)
bar.set_label(r"Posterior similarity $\mathrm{Pr}(z_{xi} = z_{x'j}\mid Y)$")
fig.savefig("gene.pdf", bbox_inches="tight")
