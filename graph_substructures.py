""""
Bayesian Learning of Graph Substructures

This Python module contains functions for posterior computation using Markov
chain Monte Carlo (MCMC) for Gaussian graphical models using Bayesian
nonparametric stochastic blockmodels as prior distribution on the graph space.
"""

import os
import time

os.environ["EXTRA_CLING_ARGS"] = "-I/usr/local/include"

import cppyy
import igraph
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import sklearn.cluster

import wwa


np.seterr(divide="ignore")  # Enable np.log(0.0)
df_0 = 3.0 # Degrees of freedom of the G-Wishart prior
s_theta = 1.0  # Prior standard deviation for theta
s_beta = 1.0  # Prior standard deviation for beta
a_alpha = 2.0  # Gamma prior on the Dirichlet concentration parameter alpha
b_alpha = a_alpha
a_nu = a_alpha  # Gamma prior on the Dirichlet concentration parameter nu
b_nu = a_nu


def get_mu(theta, beta_star, z):
    p = len(z)
    mu = np.add.outer(theta, theta)
    
    for i in range(p):
        for j in range(i + 1, p):
            if z[i] == z[j]:
                mu[i, j] += beta_star[z[i]]
    
    return mu


def MCMC_update_SBM(G, theta, beta, c, z, alpha, nu, df, rate, rng):
    """MCMC step"""
    p = G.vcount()
    _, z_index, S_size = np.unique(z, return_index=True, return_counts=True)
    _, c_index, c_size = np.unique(c, return_index=True, return_counts=True)
    theta_star = theta[c_index]
    beta_star = beta[z_index]
    mu = get_mu(theta, beta_star, z)
    prob_mu = scipy.special.ndtr(mu)
    
    # Update G
    G, log_lik = wwa.MCMC_update(G, prob_mu, df, rate, rng, get_log_lik=True)
    
    # Albert & Chib (1993) latent variable update of theta and beta
    M = len(theta_star)
    K = len(beta_star)
    S_n_pairs = S_size * (S_size - 1) // 2
    tmp = G.neighborhood(mindist=1)
    neighbors = [np.array(tmp[i], dtype=int) for i in range(p)]
    zeta_sum = np.zeros(K)
    zeta_sum_tot = 0.0
    zeta_mat = np.zeros((p, p))

    for i in range(p - 1):
        mask = np.zeros(p - i - 1, dtype=bool)
        mask[neighbors[i][neighbors[i] > i] - i - 1] = True
        
        zeta = mu[i, i + 1:] + scipy.special.ndtri(mask + (
            ~mask + (2*mask - 1)*prob_mu[i, i + 1:]
        )*(rng.random(p - i - 1) - mask))
        
        zeta_sum_tot += zeta.sum()
        k = z[i]
        zeta_sum[k] += zeta[z[i + 1:] == k].sum()
        zeta_mat[i, i + 1:] = zeta
    
    # Update theta_star
    c_pairs = c_size * (c_size - 1) // 2
    theta_var = 1.0 / (s_theta**-2 + 4*c_pairs + c_size*(p - c_size))
    
    for m in range(M):        
        theta_mean = 0.0
        
        for i in range(p - 1):
            if c[i] == m:
                for j in range(i + 1, p):
                    if c[j] == m:
                        theta_mean += 2.0 * zeta_mat[i, j]
                        
                        if z[i] == z[j]:
                            theta_mean -= 2.0 * beta[i]
                    else:
                        theta_mean += zeta_mat[i, j] - theta[j]
                        
                        if z[i] == z[j]:
                            theta_mean -= beta[i]
            else:
                for j in range(i + 1, p):
                    if c[j] == m:
                        theta_mean += zeta_mat[i, j] - theta[i]
                        
                        if z[i] == z[j]:
                            theta_mean -= beta[i]
        
        theta_star[m] = rng.normal(
            theta_var[m] * theta_mean, scale=np.sqrt(theta_var[m])
        )
        
        theta[c == m] = theta_star[m]
    
    # Update beta_star
    beta_var = 1.0 / (s_beta**-2 + S_n_pairs)
    theta_sum = np.zeros(K)
    
    for i in range(p - 1):
        for j in range(i + 1, p):
            if z[i] == z[j]:
                theta_sum[z[i]] += theta[i] + theta[j]

    beta_star = rng.normal(
        loc=beta_var * (zeta_sum - theta_sum), scale=np.sqrt(beta_var)
    )
    
    # Update c
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

        tmp = zeta_mat[i, :].sum() - theta.sum() + theta[i] \
            - (S_size[z[i]] - 1)*beta_star[z[i]]
        
        log_prob[:M] += theta_star*tmp - 0.5*(p - 1)*theta_star**2
        s_c = 1.0 / np.sqrt(p - 1 + s_theta**-2)
        m_c = s_c**2 * tmp
        log_prob[-1] += np.log(s_c) - np.log(s_theta) + 0.5*(m_c / s_c)**2
        log_prob -= log_prob.max()
        prob = np.exp(log_prob)
        c[i] = rng.choice(a=M + 1, p=prob / prob.sum())

        # try:
        #     c[i] = rng.choice(a=M + 1, p=prob / prob.sum())
        # except ValueError as e:
        #     print(e)
        
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

    # Update z
    # This is similar to Algorithm 2 of
    # Neal (2000, http://www.jstor.org/stable/1390653) and does not condition
    # on zeta. A difference from the Algorithm 2 is that the likelihood for the
    # node being considered also depends on the other nodes.
    # Note that Algorithm 8 of Neal (2000) coincides with its Algorithm 2 for
    # our model since a cluster with only one node does not have a likelihood
    # involving beta.
    for i in range(p):
        mu_ccz = np.add.outer(
            np.add.outer(theta_star, theta_star), np.append(beta_star, 0.0)
        )

        log_edge_prob_ccz = [
            scipy.special.log_ndtr(-mu_ccz),
            scipy.special.log_ndtr(mu_ccz)
        ]

        k_cur = z[i]
        z[i] = -1
        S_size[k_cur] -= 1
        log_prob = np.log(np.append(S_size, nu))  # Dirichlet process
        
        nb_ind = 0
        
        for j in range(p):
            if j == i:
                continue
            
            if nb_ind < len(neighbors[i]):
                connected = int(j == neighbors[i][nb_ind])
            else:
                connected = 0
            
            log_prob[z[j]] += log_edge_prob_ccz[connected][c[i], c[j], z[j]]

            log_prob[np.arange(K + 1) != z[j]] \
                += log_edge_prob_ccz[connected][c[i], c[j], K]
            
            nb_ind += connected
        
        log_prob -= log_prob.max()
        prob = np.exp(log_prob)
        z[i] = rng.choice(a=K + 1, p=prob / prob.sum())
        
        if z[i] == K:
            K += 1
            beta_star = np.append(beta_star, rng.normal(scale=s_beta))
            S_size = np.append(S_size, 1)
        else:
            S_size[z[i]] += 1
        
        # Remove empty clusters, if any.
        if S_size[k_cur] == 0:
            K -= 1
            z[z > k_cur] -= 1
            beta_star = np.delete(beta_star, k_cur)
            S_size = np.delete(S_size, k_cur)
    
    
    # Update alpha per Escobar & West (1995).
    log_gamma = np.log(rng.beta(alpha + 1.0, p))
    r = (a_alpha + M - 1.0) / p / (b_alpha - log_gamma)
    
    alpha = rng.gamma(
        shape=a_alpha + M if rng.random() < r / (r + 1.0) \
            else a_alpha + M - 1.0,
        scale=1.0 / (b_alpha - log_gamma)
    )
    
    
    # Update nu per Escobar & West (1995).
    log_gamma = np.log(rng.beta(nu + 1.0, p))
    r = (a_nu + K - 1.0) / p / (b_nu - log_gamma)
    
    nu = rng.gamma(
        shape=a_nu + K if rng.random() < r / (r + 1.0) else a_nu + K - 1.0,
        scale=1.0 / (b_nu - log_gamma)
    )
    
    return G, theta, beta_star[z], c, z, alpha, nu, log_lik


def MCMC_SBM(
    G_init, data, burnin, recorded, rng, verbose=True, plot=False,
    init_K1_M1=False
):
    n, p = data.shape

    for i in range(p):
        if G_init[i, i]:
            raise ValueError(
                "`G_init` contains self-loops. " \
                    + "This code assumes that there are no self-loops."
            )
    
    U = data.T @ data
    rate_0 = np.eye(p) # Rate matrix of the G-Wishart prior
    df = df_0 + n
    rate = rate_0 + U

    if init_K1_M1:
        # Initialize at single cluster cofigurations.
        z = np.zeros(p, dtype=int)
        beta = np.zeros(p)
        c = np.zeros(p, dtype=int)
        theta = np.zeros(p)
    else:
        z = np.arange(p)
        beta = rng.normal(scale=s_beta, size=p)

        # Initialize based on 3 clusters for theta using degree distribution.
        tmp = G_init.neighborhood(mindist=1)
        degree = np.array([len(tmp[i]) for i in range(p)])
        tmp = sklearn.cluster.KMeans(n_clusters=3).fit(degree.reshape(-1, 1))

        theta_star = scipy.special.ndtri(
            tmp.cluster_centers_.flatten() / (p - 1)
        ).clip(min=-4.0, max=4.0)
        
        c = tmp.labels_
        theta = theta_star[c]

        # Certain degree distributions result in fewer than k-means 3 clusters,
        # potentially leaving gaps in the values of `c`.
        c = np.unique(c, return_inverse=True)[1]

    G = G_init.copy()
    alpha = 1.0  # Concentration parameter for the Dirichlet
    nu = 1.0  # Concentration parameter for the Dirichlet
    
    # Arrays to store the MCMC chain
    edge_sum = np.zeros((p, p), dtype=int)
    theta_MCMC = np.empty((recorded, p))
    beta_MCMC = np.empty((recorded, p))
    c_MCMC = np.empty((recorded, p), dtype=int)
    z_MCMC = np.empty((recorded, p), dtype=int)
    alpha_MCMC = np.empty(recorded)
    nu_MCMC = np.empty(recorded)
    n_e_MCMC = np.empty(burnin + recorded, dtype=int)
    log_lik_MCMC = np.empty(recorded)
    
    if plot:
        t0 = time.time()
        fig, ax = plt.subplots()
        x_list = np.arange(burnin + recorded)
        line = matplotlib.lines.Line2D([], [])
        ax.add_line(line)
        ylims = [G_init.ecount() - 1, G_init.ecount() + 1]
        
        def update_plot(s, ylims, t0):
            t1 = time.time()
            
            if t1 - t0 < 1.0 and s < burnin + recorded - 1:
                return t0
            
            line.set_data(x_list[:(s + 1)], n_e_MCMC[:(s + 1)])
            ylims[0] = min(ylims[0], n_e_MCMC[s] - 1)
            ylims[1] = max(ylims[1], n_e_MCMC[s] + 1)
            
            if s > 0:
                ax.set_xlim(0, s)
            
            ax.set_ylim(ylims[0], ylims[1])
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            return t1
    
    for s in range(burnin):
        if verbose and s % 100 == 0:
            print("Burnin iteration", s, end="\r")
        
        G, theta, beta, c, z, alpha, nu, _ = MCMC_update_SBM(
            G, theta, beta, c, z, alpha, nu, df, rate, rng
        )

        n_e_MCMC[s] = G.ecount()
        
        if plot:
            t0 = update_plot(s, ylims, t0)
    
    for s in range(recorded):
        if verbose and s % 100 == 0:
            print("Iteration", s, end="\r")
        
        G, theta, beta, c, z, alpha, nu, log_lik_MCMC[s] = MCMC_update_SBM(
            G, theta, beta, c, z, alpha, nu, df, rate, rng
        )

        edge_sum += np.array(G.get_adjacency(0).data)
        theta_MCMC[s, :] = theta
        beta_MCMC[s, :] = beta
        c_MCMC[s, :] = c
        z_MCMC[s, :] = z
        alpha_MCMC[s] = alpha
        nu_MCMC[s] = nu
        n_e_MCMC[burnin + s] = G.ecount()
        
        if plot:
            t0 = update_plot(burnin + s, ylims, t0)
    
    return {
        "Edge probability": edge_sum / recorded,
        "c_MCMC": c_MCMC,
        "Group samples": z_MCMC,
        "Theta samples": theta_MCMC,
        "Beta samples": beta_MCMC,
        "Alpha samples": alpha_MCMC,
        "Nu samples": nu_MCMC,
        "n_e_MCMC": n_e_MCMC,
        "log_lik_MCMC": log_lik_MCMC,
        "last G": G
    }


def MCMC_update_Sun(z, nu, n, U, U_inv, rng):
    """
    MCMC step for the model from
    Sun et al. (2014, http://proceedings.mlr.press/v33/sun14)
    """
    p = len(U)
    _, z_index, S_size = np.unique(z, return_index=True, return_counts=True)
    K = len(S_size)

    # Update z
    for i in range(p):
        k_cur = z[i]
        z[i] = -1
        S_size[k_cur] -= 1
        log_prob = np.log(np.append(S_size, nu))  # Dirichlet process
        z_prop = z.copy()
        
        for k in range(K + 1):
            z_prop[i] = k
            V = np.equal.outer(z_prop, z_prop) * U_inv

            log_prob[k] += 0.5*n*np.linalg.slogdet(V)[1] \
                - n*np.linalg.slogdet(np.eye(p) + V@U)[1]

        
        log_prob -= log_prob.max()
        prob = np.exp(log_prob)
        z[i] = rng.choice(a=K + 1, p=prob / prob.sum())
        
        if z[i] == K:
            K += 1
            S_size = np.append(S_size, 1)
        else:
            S_size[z[i]] += 1
        
        # Remove empty clusters, if any.
        if S_size[k_cur] == 0:
            K -= 1
            z[z > k_cur] -= 1
            S_size = np.delete(S_size, k_cur)
    
    # Update nu per Escobar & West (1995).
    log_gamma = np.log(rng.beta(nu + 1.0, p))
    r = (a_nu + K - 1.0) / p / (b_nu - log_gamma)
    
    nu = rng.gamma(
        shape=a_nu + K if rng.random() < r / (r + 1.0) else a_nu + K - 1.0,
        scale=1.0 / (b_nu - log_gamma)
    )
    
    return z, nu


def MCMC_Sun(data, burnin, recorded, rng, verbose=True):
    """
    MCMC implementing the model from
    Sun et al. (2014, http://proceedings.mlr.press/v33/sun14)
    """
    n, p = data.shape

    if p > n:
        raise ValueError("`p > n` which this code does not implement.")

    U = data.T @ data
    U_inv = np.linalg.inv(U)

    # Initialize
    z = np.arange(p)
    nu = 1.0  # Concentration parameter for the Dirichlet
    
    # Arrays to store the MCMC chain
    z_MCMC = np.empty((recorded, p), dtype=int)
    nu_MCMC = np.empty(recorded)
    
    for s in range(burnin):
        if verbose and s % 100 == 0:
            print("Burnin iteration", s, end="\r")
        
        z, nu = MCMC_update_Sun(z, nu, n, U, U_inv, rng)
    
    for s in range(recorded):
        if verbose and s % 100 == 0:
            print("Iteration", s, end="\r")
        
        z, nu = MCMC_update_Sun(z, nu, n, U, U_inv, rng)
        z_MCMC[s, :] = z
        nu_MCMC[s] = nu
    
    return {
        "Group samples": z_MCMC,
        "Nu samples": nu_MCMC
    }



cppyy.cppdef("""
#include <cmath>
#include <iostream>
#include <vector>

// This code was tested using Blaze version 3.8.0
// (https://bitbucket.org/blaze-lib/blaze/src/master/) with the fix from this
// pull request: https://bitbucket.org/blaze-lib/blaze/pull-requests/46.
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Column.h>
#include <blaze/math/Columns.h>
#include <blaze/math/Row.h>
#include <blaze/math/Rows.h>
#include <blaze/math/Elements.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/Band.h>

// This code was tested using igraph version 0.8.5 (https://igraph.org).
#include <igraph/igraph.h>


blaze::DynamicMatrix<double> inv_pos_def(blaze::DynamicMatrix<double> mat) {
    blaze::invert<blaze::byLLH>(mat);
    return mat;
}


template <class T>
double log_det(T& mat) {
    // Log of the matrix determinant of `mat`
    blaze::DynamicMatrix<double> L;
    llh(mat, L);
    return 2.0 * sum(log(diagonal(L)));
}


blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > gwish_mode_inv(
    igraph_t* G_ptr, double df,
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& rate
) {
    /*
    Find the inverse of the mode of a G-Wishart distribution.
    
    `G_ptr` is a pointer the graph to which the precision is constraint.

    `df` is the degrees of freedom of the distribution.

    `rate` is the rate or inverse scale matrix of the distribution and must be
    symmetric positive definite.
    
    The notation in this function follows Section 2.4 in
    Lenkoski (2013, arXiv:1304.1350v1).
    
    The optimization procedure is presented in Algorithm 17.1 of
    the Elements of Statistical Learning by Hastie et al.
    
    Compare Equation 7 from
    https://proceedings.neurips.cc/paper/2009/hash/a1519de5b5d44b31a01de013b9b51a80-Abstract.html
    with Equation 17.11 of the Elements of Statistical Learning by
    Hastie et al. to understand why we set `Sigma = rate / (df - 2.0)`.
    */
    int p = rate.rows();
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        Sigma(rate / (df - 2.0)), W(Sigma);  // Step 1

    // Inspired by C++ code from the R package BDgraph
    std::vector<std::vector<double> > neighbors(p);
    std::vector<blaze::DynamicVector<double> > Sigma_N(p);
    // Avoid recomputing the neighbors for each iteration:
    igraph_vector_ptr_t igraph_neighbors;
    igraph_vector_ptr_init(&igraph_neighbors, p);

    igraph_neighborhood(
        G_ptr, &igraph_neighbors, igraph_vss_all(), 1, IGRAPH_ALL, 1
    );

    for (int i = 0; i < p; i++) {
        igraph_vector_t* N_ptr
            = (igraph_vector_t*) igraph_vector_ptr_e(&igraph_neighbors, i);

        neighbors[i].resize(igraph_vector_size(N_ptr));

        for (int j = 0; j < neighbors[i].size(); j++)
            neighbors[i][j] = igraph_vector_e(N_ptr, j);

        Sigma_N[i] = elements(column(Sigma, i), neighbors[i]);
    }

    IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR(
        &igraph_neighbors, igraph_vector_destroy
    );

    igraph_vector_ptr_destroy_all(&igraph_neighbors);
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > W_previous(p);
    blaze::DynamicMatrix<double, blaze::columnMajor> W_N;
    blaze::DynamicMatrix<double, blaze::rowMajor> W_NN;
    blaze::DynamicVector<double> W_beta_hat(p), beta_star;
    
    for (int i = 0; i < 10000; i++) {
        W_previous = W;
        
        for (int j = 0; j < p; j++) {
            if (neighbors[j].size() == 0) {
                // This only happens if the graph `G_ptr` is not connected.
                W_beta_hat = 0.0;
            } else if (neighbors[j].size() == p - 1) {
                subvector(W_beta_hat, 0, j) = subvector(Sigma_N[j], 0, j);

                subvector(W_beta_hat, j + 1, p - j - 1)
                    = subvector(Sigma_N[j], j, p - j - 1);
            } else {
                W_N = columns(W, neighbors[j]);
                W_NN = rows(W_N, neighbors[j]);
                solve(declsym(W_NN), beta_star, Sigma_N[j]);
                W_beta_hat = W_N * beta_star;
            }

            double W_jj = W(j, j);
            column(W, j) = W_beta_hat;
            // The next line is not need as Blaze enforces the symmetry of `W`.
            // row(W, j) = trans(W_beta_hat);
            W(j, j) = W_jj;
        }

        // 1e-8 is consistent with BDgraph
        if (blaze::mean(blaze::abs(W - W_previous)) < 1e-8) return W;
    }

    std::cout << "`gwish_mode_inv` failed to converge." << std::endl;
    return W;
}


blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > hessian(
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& K_inv,
    igraph_t G_V, double df
) {
    // Compute the Hessian divided by `-0.5 * (df - 2.0)`.
    int n_e = igraph_ecount(&G_V);
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > H(n_e);

    for (int a = 0; a < n_e; a++) {
        int i = IGRAPH_FROM(&G_V, a), j = IGRAPH_TO(&G_V, a);

        for (int b = a; b < n_e; b++) {
            int l = IGRAPH_FROM(&G_V, b), m = IGRAPH_TO(&G_V, b);

            if (i == j) {
                if (l == m) {
                    H(a, b) = std::pow(K_inv(i, l), 2);
                } else {
                    H(a, b) = 2.0 * K_inv(i, l) * K_inv(i, m);
                }
            } else {
                if (l == m) {
                    H(a, b) = 2.0 * K_inv(i, l) * K_inv(j, l);
                } else {
                    H(a, b) = 2.0 * (
                        K_inv(i, l)*K_inv(j, m) + K_inv(i, m)*K_inv(j, l)
                    );
                }
            }
        }
    }

    return H;
}


double log_gwish_norm_laplace(
    igraph_t* G_ptr, double df,
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& rate,
    bool diag = false
) {
    /*
    Log of Laplace approximation of G-Wishart normalization constant
    
    Log of the Laplace approximation of the normalization constant of the
    G-Wishart distribution outlined by
    Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)

    `diag` indicates whether the full Hessian matrix should be used
    (`diag = false`) or only its diagonal (`diag = true`). The latter is faster
    but less accurate per Moghaddam et al. (2009,
    https://papers.nips.cc/paper/2009/hash/a1519de5b5d44b31a01de013b9b51a80-Abstract.html).
    */
    int p = rate.rows(), n_e = igraph_ecount(G_ptr);

    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        K_inv = gwish_mode_inv(G_ptr, df, rate);

    blaze::DynamicMatrix<double> K = inv_pos_def(K_inv);
    double log_det_H, h = -0.5 * (trace(K * rate) - (df - 2.0)*log_det(K));
    
    if (diag) {
        log_det_H = 2.0*sum(log(diagonal(K_inv))) + n_e*std::log(2.0);

        for (int vid = 0; vid < n_e; vid++) {
            int i = IGRAPH_FROM(G_ptr, vid), j = IGRAPH_TO(G_ptr, vid);

            log_det_H +=
                std::log(K_inv(i, i)*K_inv(j, j) + std::pow(K_inv(i, j), 2));
        }
    } else {
        // Create graph `G_V` which equals `G` plus all self-loops such that
        // its edge set coincides with Equation 2.1 of
        // Lenkoski and Dobra (2011).
        igraph_t G_V;
        igraph_copy(&G_V, G_ptr);
        igraph_vector_t edge_list;
        igraph_vector_init(&edge_list, 2 * p);

        for (int i = 0; i < p; i++) {
            VECTOR(edge_list)[2*i] = i;
            VECTOR(edge_list)[2*i + 1] = i;
        }

        igraph_add_edges(&G_V, &edge_list, 0);
        igraph_vector_destroy(&edge_list);

        blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
            H = hessian(K_inv, G_V, df);

        igraph_destroy(&G_V);
        log_det_H = log_det(H);
    }

    // The sign of the Hessian `-0.5 * (df - 2.0) * H` is flipped compared to
    // Lenkoski and Dobra (2011, Section 4). I think that this is correct as
    // |Hessian| can be negative while |-Hessian| cannot.
    return h + 0.5*(p + n_e)*std::log(2.0 * M_PI)
        - 0.5*((p + n_e)*std::log(0.5 * (df - 2.0)) + log_det_H);
}


double log_gwish_norm_laplace_cpp(
    igraph_t* G_ptr, double df, double* rate_in, bool diag = false
) {
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        rate(igraph_vcount(G_ptr), rate_in);

    return log_gwish_norm_laplace(G_ptr, df, rate, diag);
}
""")

# Reduce Python overhead:
# If the next line fails, then the compilations of the C++ code might have
# failed.
log_gwish_norm_laplace_cpp = cppyy.gbl.log_gwish_norm_laplace_cpp


def log_gwish_norm_laplace(G, df, rate=None, diag=True): 
    """
    Log of Laplace approximation of G-Wishart normalization constant
    
    Log of the Laplace approximation of the normalization constant of the
    G-Wishart distribution outlined by
    Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)
    """
    if rate is None:
        # If the rate matrix is the identity matrix (I_p), then the mode of the
        # G-Wishart distribution is (df - 2) * I_p such that the Laplace
        # approximation simplifies.
        p = G.vcount()
        n_e = G.ecount()
        
        return 0.5 * (
            (p*(df - 1.0) + n_e)*np.log(df - 2.0) - p*(df - 2.0) \
                + p*np.log(4.0 * np.pi) + n_e*np.log(2.0 * np.pi)
        )
    
    return log_gwish_norm_laplace_cpp(G.__graph_as_capsule(), df, rate, diag)


large_int = 2**63


def random_seed(rng):
    # The following is faster than `rng.integers(large_int)`.
    return int(large_int * rng.random())


def log_likelihood(G, df, rate, diag=True):
    return log_gwish_norm_laplace(G, df, rate, diag) \
        - log_gwish_norm_laplace(G, df_0)


def sample_edge(G, rng):
    """Sample an edge from graph `G`."""
    return G.es[rng.choice(G.ecount())].tuple


def proposal_G_es(n_e_tilde, n_e, max_e_z):
    """
    Proposal transition probability from `G` to `G_tilde` based on edge counts
    """
    if n_e == 0 or n_e == max_e_z:
        return 1.0 / max_e_z
    elif n_e > n_e_tilde:
        return 0.5 / n_e
    else:
        return 0.5 / (max_e_z - n_e)


def MCMC_update_SICS(G, edge_prob, z, nu, df, rate, rng, diag=True):
    """MCMC step for the Southern Italian community structure (SICS)"""
    p = G.vcount()
    max_e = p * (p - 1) // 2
    _, z_index, S_size = np.unique(z, return_index=True, return_counts=True)
    K = len(S_size)
    log_lik_cur = log_likelihood(G, df, rate, diag)
    min_e_z = np.sum(S_size * (S_size - 1) // 2)
    n_e = G.ecount() - min_e_z
    max_e_z = max_e - min_e_z
    
    # Update `edge_prob`
    edge_prob = rng.beta(a=1 + n_e, b=1 + max_e_z - n_e)
    
    if max_e_z != 0:
        # Decide whether to propose an edge addition or removal.
        if n_e == 0:
            add = True
        elif n_e == max_e_z:
            add = False
        else:
            add = rng.random() < 0.5

        # Pick edge to add or remove.
        if add:
            e = sample_edge(~G, rng=rng)
        else:
            G_tmp = G.copy()
            
            for i in range(p):
                for j in range(i + 1, p):
                    if z[i] == z[j]:
                        G_tmp[i, j] = False
            
            e = sample_edge(G_tmp, rng=rng)

        G_tilde = G.copy()
        G_tilde[e] = True if add else False
        log_lik_prop = log_likelihood(G_tilde, df, rate, diag)
        
        if np.log(rng.random()) < log_lik_prop - log_lik_cur \
            + np.log(proposal_G_es(n_e, n_e + 2*add - 1, max_e_z)) \
            - np.log(proposal_G_es(n_e + 2*add - 1, n_e, max_e_z)) \
            + (2*add - 1)*(np.log(edge_prob) - np.log(1.0 - edge_prob)):
                G = G_tilde
                log_lik_cur = log_lik_prop
    
    # Update z
    # This is similar to Algorithm 2 of
    # Neal (2000, http://www.jstor.org/stable/1390653) and does not condition
    # on zeta. A difference from the Algorithm 2 is that the likelihood for the
    # node being considered also depends on the other nodes.
    # Note that Algorithm 8 of Neal (2000) coincides with its Algorithm 2 for
    # our model since a cluster with only one node does not have a likelihood
    # involving beta.
    for i in range(p):
        k_cur = z[i]
        k_prop = rng.choice(K + (S_size[k_cur] != 1))
        
        if k_prop == k_cur:
            continue
        
        # Dirichlet process
        if S_size[k_cur] == 1:
            log_prob_cur = np.log(nu)
        else:
            log_prob_cur = np.log(S_size[k_cur] - 1)
        
        if k_prop == K:
            log_prob_prop = np.log(nu)
        else:
            log_prob_prop = np.log(S_size[k_prop])

        G_tilde = G.copy()
            
        for j in range(p):
            if j ==i:
                continue
            
            if z[j] == k_cur:
                G_tilde[i, j] = rng.random() < edge_prob
            
            if z[j] == k_prop:
                G_tilde[i, j] = True
        
        log_lik_prop = log_likelihood(G_tilde, df, rate, diag)
        
        if np.log(rng.random()) > \
            log_prob_prop + log_lik_prop - log_prob_cur - log_lik_cur:
                continue
        
        z[i] = k_prop
        S_size[k_cur] -= 1
        G = G_tilde
        log_lik_cur = log_lik_prop
        
        if k_prop == K:
            K += 1
            S_size = np.append(S_size, 1)
        else:
            S_size[k_prop] += 1

        # Remove empty clusters, if any.
        if S_size[k_cur] == 0:
            K -= 1
            z[z > k_cur] -= 1
            S_size = np.delete(S_size, k_cur)
    
    # Update nu per Escobar & West (1995).
    log_gamma = np.log(rng.beta(nu + 1.0, p))
    r = (a_nu + K - 1.0) / p / (b_nu - log_gamma)
    
    nu = rng.gamma(
        shape=a_nu + K if rng.random() < r / (r + 1.0) else a_nu + K - 1.0,
        scale=1.0 / (b_nu - log_gamma)
    )
    
    # Compute the log-likelihood for a specific K.
    K = wwa.rgwish(G, df, rate, rng)

    log_lik = 0.5*(df - df_0)*(
        np.linalg.slogdet(K)[1] - p*np.log(2.0 * np.pi)
    ) - np.trace(K @ (rate - np.eye(p)))

    return G, edge_prob, z, nu, log_lik


def MCMC_SICS(
    G_init, data, burnin, recorded, rng, verbose=True, plot=False, diag=False
):
    n, p = data.shape
    U = data.T @ data
    rate_0 = np.eye(p)  # Rate matrix of the G-Wishart prior
    df = df_0 + n
    rate = rate_0 + U

    # Initialize
    G = G_init.copy()
    z = np.arange(p)
    edge_prob = 0.5
    nu = 1.0  # Concentration parameter for the Dirichlet

    edge_count = np.zeros((p, p), dtype=int)
    edge_prob_MCMC = np.empty(recorded)
    z_MCMC = np.empty((recorded, p), dtype=int)
    nu_MCMC = np.empty(recorded)
    n_e = np.empty(burnin + recorded, dtype=int)
    log_lik_MCMC = np.empty(recorded)
    
    if plot:
        t0 = time.time()
        fig, ax = plt.subplots()
        x_list = np.arange(burnin + recorded)
        line = matplotlib.lines.Line2D([], [])
        ax.add_line(line)
        ylims = [G_init.ecount() - 1, G_init.ecount() + 1]
        
        def update_plot(s, ylims, t0):
            t1 = time.time()
            
            if t1 - t0 < 1.0 and s < burnin + recorded - 1:
                return t0
            
            t0 = t1
            line.set_data(x_list[:(s + 1)], n_e[:(s + 1)])
            ylims[0] = min(ylims[0], n_e[s] - 1)
            ylims[1] = max(ylims[1], n_e[s] + 1)
            
            if s > 0:
                ax.set_xlim(0, s)
            
            ax.set_ylim(ylims[0], ylims[1])
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            return t1
    
    for s in range(burnin):
        if verbose and s % 100 == 0:
            print("Burnin iteration", s, end="\r")
        
        G, edge_prob, z, nu, _ = MCMC_update_SICS(
            G, edge_prob, z, nu, df, rate, rng, diag
        )

        n_e[s] = G.ecount()
        
        if plot:
            t0 = update_plot(s, ylims, t0)
    
    for s in range(recorded):
        if verbose and s % 100 == 0:
            print("Iteration", s, end="\r")
        
        G, edge_prob, z, nu, log_lik_MCMC[s] = MCMC_update_SICS(
            G, edge_prob, z, nu, df, rate, rng, diag
        )

        edge_count += np.array(G.get_adjacency(0).data)
        z_MCMC[s, :] = z
        nu_MCMC[s] = nu
        edge_prob_MCMC[s] = edge_prob
        n_e[burnin + s] = G.ecount()
        
        if plot:
            t0 = update_plot(burnin + s, ylims, t0)
    
    return {
        "Edge probability": edge_count / recorded,
        "Group samples": z_MCMC,
        "Nu samples": nu_MCMC,
        "edge_prob_MCMC": edge_prob_MCMC,
        "n_e": n_e,
        "log_lik_MCMC": log_lik_MCMC,
        "last G": G
    }
