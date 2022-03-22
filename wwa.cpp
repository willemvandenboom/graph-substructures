#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// This code was tested using Blaze version 3.8.0
// (https://bitbucket.org/blaze-lib/blaze/src/master/) with the fix from this
// pull request: https://bitbucket.org/blaze-lib/blaze/pull-requests/46.
// Avoid nested parallization error:
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/LowerMatrix.h>
#include <blaze/math/UpperMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Column.h>
#include <blaze/math/Columns.h>
#include <blaze/math/Row.h>
#include <blaze/math/Rows.h>
#include <blaze/math/Elements.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>

// The distributions in `<random>` are not portable. That is, they do not
// yield the same random numbers on different machines. Therefore, we use the
// distributions from Boost, which are protable and sometimes also faster.
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// This code was tested using igraph version 0.8.5 (https://igraph.org).
#include <igraph/igraph.h>


// C++'s RNGs are not very fast. Therefore, I use the RNG from
// https://gist.github.com/martinus/c43d99ad0008e11fcdbf06982e25f464:
// extremely fast random number generator that also produces very high quality random.
// see PractRand: http://pracrand.sourceforge.net/PractRand.txt
class sfc64 {
  public:
    using result_type = uint64_t;

    static constexpr uint64_t(min)() { return 0; }
    static constexpr uint64_t(max)() { return UINT64_C(-1); }

    sfc64() : sfc64(std::random_device{}()) {}

    explicit sfc64(uint64_t seed) : m_a(seed), m_b(seed), m_c(seed), m_counter(1) {
        for (int i = 0; i < 12; ++i) {
            operator()();
        }
    }

    uint64_t operator()() noexcept {
        auto const tmp = m_a + m_b + m_counter++;
        m_a = m_b ^ (m_b >> right_shift);
        m_b = m_c + (m_c << left_shift);
        m_c = rotl(m_c, rotation) + tmp;
        return tmp;
    }

  private:
    template <typename T> T rotl(T const x, int k) { return (x << k) | (x >> (8 * sizeof(T) - k)); }

    static constexpr int rotation = 24;
    static constexpr int right_shift = 11;
    static constexpr int left_shift = 3;
    uint64_t m_a;
    uint64_t m_b;
    uint64_t m_c;
    uint64_t m_counter;
};


blaze::DynamicMatrix<double> inv_pos_def(blaze::DynamicMatrix<double> mat) {
    blaze::invert<blaze::byLLH>(mat);
    return mat;
}


template <class T>
auto submatrix_view(
    blaze::DynamicMatrix<double>& mat, std::vector<T>& ind_row,
    std::vector<T>& ind_col
) {
    return columns(rows(mat, ind_row), ind_col);
}


template <class T>
auto submatrix_view_square(
    blaze::DynamicMatrix<double>& mat, std::vector<T>& ind
) {
    return submatrix_view(mat, ind, ind);
}


blaze::DynamicMatrix<double> submatrix(
    blaze::DynamicMatrix<double>& mat, std::vector<double>& ind
) {
    return blaze::DynamicMatrix<double>(columns(rows(mat, ind), ind));
}


template <class T1, typename T2>
void submatrix_assign(
    blaze::DynamicMatrix<double>& mat_out,
    T1& mat_in, std::vector<T2>& ind_row, std::vector<T2>& ind_col,
    bool add = false
) {
    if (mat_in.rows() != ind_row.size() or mat_in.columns() != ind_col.size())
        throw std::runtime_error(
            "Size of `mat_in` does not match `ind_row` and `ind_col`."
        );

    if (add) {
        for (int i = 0; i < ind_row.size(); i++)
            for (int j = 0; j < ind_col.size(); j++)
                mat_out(ind_row[i], ind_col[j]) += mat_in(i, j);

        return;
    }

    for (int i = 0; i < ind_row.size(); i++)
        for (int j = 0; j < ind_col.size(); j++)
                mat_out(ind_row[i], ind_col[j]) = mat_in(i, j);
}


template <class T1, typename T2>
void submatrix_assign_square(
    blaze::DynamicMatrix<double>& mat_out, T1& mat_in, std::vector<T2>& ind,
    bool add = false
) {
    submatrix_assign(mat_out, mat_in, ind, ind, add);
}


bool is_complete(igraph_t* G_ptr) {
    int p = igraph_vcount(G_ptr);
    return igraph_ecount(G_ptr) == p * (p - 1) / 2;
};


blaze::UpperMatrix<blaze::DynamicMatrix<double> > rwish_identity_chol(
    int p, double df, sfc64& rng
) {
    blaze::UpperMatrix<blaze::DynamicMatrix<double> > Phi(p);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);
    df += p - 1;
    
    // Generate the upper-triangular Cholesky decompositon of a standard
    // Wishart random variable.
    for (int i = 0; i < p; i++) {
        boost::random::chi_squared_distribution<> rchisq(df - i);
        Phi(i, i) = std::sqrt(rchisq(rng));
        for (int j = i + 1; j < p; j++) Phi(i, j) = rnorm(rng);
    }
    
    return Phi;
}


blaze::DynamicMatrix<double> rwish_identity(int p, double df, sfc64& rng) {
    /*
    Sample a `p` by `p` matrix from a Wishart distribution with `df` degrees of
    freedom with an identity rate matrix.
    */

    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        Phi = rwish_identity_chol(p, df, rng);
    
    return declsym(trans(Phi) * Phi);
}


blaze::DynamicMatrix<double> rwish(
    double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        Phi_identity = rwish_identity_chol(rate.rows(), df, rng);

    blaze::LowerMatrix<blaze::DynamicMatrix<double> > chol;
    llh(rate, chol);
    blaze::invert(chol);
    blaze::DynamicMatrix<double> Phi = Phi_identity * chol;
    return declsym(trans(Phi) * Phi);
}


blaze::DynamicMatrix<double> rgwish_empty_identity(
    int p, double df, sfc64& rng
) {
    blaze::DynamicMatrix<double> K(p, p, 0.0);
    boost::random::chi_squared_distribution<> rchisq(df);
    for (int i = 0; i < p; i++) K(i, i) = rchisq(rng);
    return K;
}


blaze::DynamicMatrix<double> rgwish_empty(
    int p, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    blaze::DynamicMatrix<double> K(p, p, 0.0);
    boost::random::chi_squared_distribution<> rchisq(df);
    for (int i = 0; i < p; i++) K(i, i) = rchisq(rng) / rate(i, i);
    return K;
}


blaze::DynamicMatrix<double> rginvwish_L_body(
    igraph_t* G_ptr, blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > W
) {
    // This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        Sigma = inv_pos_def(W);

    if (is_complete(G_ptr)) return Sigma;
    int p = W.rows();
    W = Sigma;  // Step 1

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
    // blaze::DynamicMatrix<double> W_NN;
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

    std::cout << "`rgwish_L` failed to converge." << std::endl;
    return W;
}


blaze::DynamicMatrix<double> rginvwish_L(
    igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    /*
    Sample from the G-inverse-Wishart using the Lenkoski method.
    
    `rate` is the inverse of the scale matrix of the Wishart distribution.
    This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    */
    return rginvwish_L_body(G_ptr, rwish(df, rate, rng));
}


blaze::DynamicMatrix<double> rgwish_L(
    igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    /*
    Sample from the G-Wishart using the Lenkoski method.
    
    `rate` is the inverse of the scale matrix of the Wishart distribution.
    This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    */
    if (is_complete(G_ptr)) return rwish(df, rate, rng);
    return inv_pos_def(rginvwish_L(G_ptr, df, rate, rng));
}


blaze::DynamicMatrix<double> rginvwish_L_identity(
    igraph_t* G_ptr, double df, sfc64& rng
) {
    return rginvwish_L_body(
        G_ptr, rwish_identity(igraph_vcount(G_ptr), df, rng)
    );
}


blaze::DynamicMatrix<double> rgwish_L_identity(
    igraph_t* G_ptr, double df, sfc64& rng
) {
    if (is_complete(G_ptr))
        return rwish_identity(igraph_vcount(G_ptr), df, rng);

    return inv_pos_def(rginvwish_L_identity(G_ptr, df, rng));
}


std::vector<std::vector<int> > igraph2adjlist(igraph_t* G_ptr) {
    // Put the graph in adjacency list format using `std::vector`.
    // The resulting adjacency lists are sorted.
    int p = igraph_vcount(G_ptr);
    igraph_adjlist_t G_adjlist;
    igraph_adjlist_init(G_ptr, &G_adjlist, IGRAPH_ALL);
    std::vector<std::vector<int> > adjlist(p, std::vector<int>(0));

    for (int i = 0; i < p; i++) {
        igraph_vector_int_t* N_G_ptr = igraph_adjlist_get(&G_adjlist, i);
        adjlist[i].resize(igraph_vector_int_size(N_G_ptr));

        for (int j = 0; j < adjlist[i].size(); j++)
            adjlist[i][j] = VECTOR(*N_G_ptr)[j];
    }

    igraph_adjlist_destroy(&G_adjlist);
    return adjlist;
}


auto decompose_graph(igraph_t* G_ptr) {
    /*
    Compute the prime subgraphs and clique minimal separators of the graph in a
    perfect ordering.
    
    This follows the MCSM-Atom-Tree algorithm from
    Berry et al. (2014, page 9, doi:10.1016/j.dam.2014.05.030).
    */
    // G_prime is the subgraph of G with nodes that still need to be covered.
    // H is the minimally triangulated graph.
    // We store G_prime and H in adjacency list format as that is more
    // efficient for our purposes.
    std::vector<std::vector<int> > G_prime_adjlist = igraph2adjlist(G_ptr),
        H_adjlist = G_prime_adjlist;

    int n = G_prime_adjlist.size(), prev_card = 0, new_card, x, y, s, x_index;
    
    // We use `double` instead of `int` in the next 2 lines
    // because `igraph_vector_view` doesn't accept integers.  // This point is moot now that I have discovered `igraph_vector_int_view`.
    std::vector<std::vector<double> > primes(0), seps(0);
    std::vector<double> Madj;
    std::vector<std::vector<int> > reach(n, std::vector<int>(0));

    // `V_prime` keeps track of which nodes have not yet been covered. We also
    // keep track of the number of neighbors in H that have already been
    // visited. This is referred to as `label` in
    // Berry et al. (2010, doi:10.3390/a3020197).
    // `order` is the perfect elimination ordering of H.
    // `ato` is the atom index for each node.
    std::vector<int> V_prime(n), label(n, 0), order(n, 0), ato(n), Y;
    std::vector<bool> reached(n);
    igraph_bool_t clique;
    std::iota(V_prime.begin(), V_prime.end(), 0);

    for (int i = n - 1; true; i--) {
        // Instead of `argmax`, we could keep track of the maximum as we update `label`.
        // x = V_prime[np.argmax(label[V_prime])]
        new_card = -1;

        for (int j = 0; j < V_prime.size(); j++) {
            if (label[V_prime[j]] > new_card) {
                x_index = j;
                new_card = label[V_prime[j]];
            }
        }

        x = V_prime[x_index];
        order[x] = i;
        // Ensure that `H_adjlist[x]` is sorted.
        std::sort(H_adjlist[x].begin(), H_adjlist[x].end());

        if (new_card <= prev_card) {  // Begin new clique
            if (new_card == 0) {
                // This differs from Berry et al. (2014).
                // I think this is correct.
                s = primes.size(); 
                primes.push_back({});
            } else {
                Madj.clear();

                for (int j = 0; j < H_adjlist[x].size(); j++)
                    if (order[H_adjlist[x][j]] > i)
                        Madj.push_back(H_adjlist[x][j]);

                clique = true;

                for (int j = 0; j < Madj.size(); j++) {
                    for (int k = 0; k < j; k++) {
                        igraph_are_connected(G_ptr, Madj[j], Madj[k], &clique);
                        if (not clique) break;
                    }

                    if (not clique) break;
                }

                if (clique) {
                    // This differs from Berry et al. (2014).
                    // I think this is correct.
                    s = primes.size();
                    primes.push_back(Madj);
                    seps.push_back(Madj);
                } else {
                    // s = ato[Madj[np.argmin(order[Madj])]]
                    int tmp = n;

                    for (int y : Madj) if (order[y] < tmp) {
                        s = ato[y];
                        tmp = order[y];
                    }
                }

            }

        }

        ato[x] = s;
        primes[s].push_back(x);

        if (i == 0) {
            // Ensure the vectors are sorted before returning.
            for (std::vector<double>& pr : primes)
                std::sort(pr.begin(), pr.end());

            for (std::vector<double>& se : seps)
                std::sort(se.begin(), se.end());

            return std::make_tuple(primes, seps);
        }

        // The following part of the algorithm follows the ideas in the MCS-M+
        // algorithm from Berry et al. (2010, page 208, doi:10.3390/a3020197).
        for (int j = 0; j < n; j++) {
            reached[j] = false;
            reach[j].clear();
        }

        reached[x] = true;
        Y.clear();
        // Delete elements not in `V_prime` from `H_adjlist[x]`.
        // We exploit the fact that `V_prime` and `H_adjlist[x]` are sorted.
        std::vector<int, std::allocator<int> >::iterator
            vec_ptr = V_prime.end() - 1;

        for (long j = H_adjlist[x].size() - 1; j >= 0; j--) {
            while (
                *vec_ptr > H_adjlist[x][j] and vec_ptr != V_prime.begin()
            ) vec_ptr--;
            
            if (H_adjlist[x][j] != *vec_ptr)
                H_adjlist[x].erase(H_adjlist[x].begin() + j);
        }

        for (int j = 0; j < H_adjlist[x].size(); j++) {
            int node = H_adjlist[x][j];
            reached[node] = true;
            reach[label[node]].push_back(node);
        }

        for (int j = 0; j < n; j++) while (reach[j].size() != 0) {
            y = reach[j].back();
            reach[j].pop_back();

            for (int k = 0; k < G_prime_adjlist[y].size(); k++) {
                int z = G_prime_adjlist[y][k];

                if (not reached[z]) {
                    reached[z] = true;

                    if (label[z] > j) {
                        Y.push_back(z);
                        reach[label[z]].push_back(z);
                    } else {
                        reach[j].push_back(z);
                    }
                }
            }
        }
        
        for (int y : Y) H_adjlist[y].push_back(x);

        // Delete vertex `x` from G_prime.
        // We exploit that `G_prime_adjlist[j]` is sorted.
        for (int j = 0; j < n; j++) {
            std::vector<int, std::allocator<int> >::iterator
                tmp = std::lower_bound(
                    G_prime_adjlist[j].begin(), G_prime_adjlist[j].end(), x
                );

            if (tmp != G_prime_adjlist[j].end() and *tmp == x)
                G_prime_adjlist[j].erase(tmp);
        }

        // Ensure that x isn't considered anymore.
        V_prime.erase(V_prime.begin() + x_index);
        
        // `label` changes as `V_prime` has changed.
        for (int j = 0; j < H_adjlist[x].size(); j++) label[H_adjlist[x][j]]++;
        for (int y : Y) label[y]++;
        prev_card = new_card;
    }
}


igraph_t get_subgraph(igraph_t* G_ptr, std::vector<double>& vec) {
    igraph_t G_sub;
    igraph_vector_t igraph_vec;

    igraph_induced_subgraph(G_ptr, &G_sub, igraph_vss_vector(
        igraph_vector_view(&igraph_vec, vec.data(), vec.size())
    ), IGRAPH_SUBGRAPH_AUTO);

    return G_sub;
}


std::vector<std::vector<double> > get_components(igraph_t* G_ptr) {
    /*
    Split the graph into its connected components.

    This function returns `double`s instead of `int`s because igraph works with
    reals rather than integers.
    */
    int n_c, p = igraph_vcount(G_ptr);
    igraph_vector_t membership, csize;
    igraph_vector_init(&membership, p);
    igraph_vector_init(&csize, 0);
    igraph_clusters(G_ptr, &membership, &csize, &n_c, IGRAPH_STRONG);
    std::vector<std::vector<double> > components(n_c, std::vector<double>(0));
    for (int i = 0; i < p; i++) components[VECTOR(membership)[i]].push_back(i);
    
    // Release memory.
    igraph_vector_destroy(&membership);
    igraph_vector_destroy(&csize);

    return components;
}


blaze::DynamicMatrix<double> rgwish_identity(
    igraph_t* G_ptr, double df, sfc64& rng, int& max_prime
) {
    max_prime = 1;  // Number of nodes of the largest prime component
    int p = igraph_vcount(G_ptr);
    if (igraph_ecount(G_ptr) == 0) return rgwish_empty_identity(p, df, rng);
    if (is_complete(G_ptr)) return rwish_identity(p, df, rng);

    // Split the graph into its connected components.
    std::vector<std::vector<double> > components = get_components(G_ptr);
    blaze::DynamicMatrix<double> K(p, p, 0.0);
    boost::random::chi_squared_distribution<> rchisq(df);

    for (std::vector<double> component : components) {
        int p_c = component.size();

        if (p_c == 1) {
            int i = component[0];
            K(i, i) = rchisq(rng);
            continue;
        }

        igraph_t G_c = get_subgraph(G_ptr, component);
        
        if (is_complete(&G_c)) {
            if (p_c > max_prime) max_prime = p_c;
            blaze::DynamicMatrix<double> K_c = rwish_identity(p_c, df, rng);
            submatrix_assign_square(K, K_c, component);
            igraph_destroy(&G_c);  // Release memory.
            continue;
        }

        auto [primes, seps] = decompose_graph(&G_c);
        if (primes[0].size() > max_prime) max_prime = primes[0].size();

        if (seps.size() == 0) {  // `G_c` is a prime graph.
            blaze::DynamicMatrix<double>
                K_c = rgwish_L_identity(&G_c, df, rng);
            
            submatrix_assign_square(K, K_c, component);
            igraph_destroy(&G_c);  // Release memory.
            continue;
        }

        // Start of the generation of the G-Wishart with the first prime.
        igraph_t G_prime = get_subgraph(&G_c, primes[0]);

        blaze::DynamicMatrix<double> K_c(p_c, p_c, 0.0), Sigma(p_c, p_c, 0.0),
                K_prime, K_prime_inv;

        if (is_complete(&G_prime)) {
            blaze::DynamicMatrix<double>
                Phi = rwish_identity_chol(primes[0].size(), df, rng);

            K_prime = declsym(trans(Phi) * Phi);
            blaze::invert(Phi);  //blaze::invert(declupp(Phi));
            K_prime_inv = declsym(Phi * trans(Phi));
        } else {
            K_prime_inv = rginvwish_L_identity(&G_prime, df, rng);
            K_prime = inv_pos_def(K_prime_inv);
        }

        igraph_destroy(&G_prime);

        // Equation 7 of Carvalho et al. (2007, doi:10.1093/biomet/asm056)
        submatrix_assign_square(K_c, K_prime, primes[0]);
        submatrix_assign_square(Sigma, K_prime_inv, primes[0]);

        for (int i = 0; i < seps.size(); i++) {
            if (primes[i + 1].size() > max_prime)
                max_prime = primes[i + 1].size();

            G_prime = get_subgraph(&G_c, primes[i + 1]);
            K_prime = rgwish_L_identity(&G_prime, df, rng);
            igraph_destroy(&G_prime);

            // R = list(set(prime) - set(sep))
            // `sep_sub` and `R_sub` contain the same indices as `sep`
            // and `R` but then relative to the current prime subgraph.
            // We exploit that seps[i] and primes[i + 1] are sorted.
            std::vector<double> sep_sub(seps[i].size()),
                R(primes[i + 1].size() - sep_sub.size()), R_sub(R.size());

            int sep_index = 0;

            for (int j = 0; j < primes[i + 1].size(); j++) if (
                sep_index < seps[i].size()
                    and primes[i + 1][j] == seps[i][sep_index]
            ) {
                sep_sub[sep_index] = j;
                sep_index++;
            } else {
                R[j - sep_index] = primes[i + 1][j];
                R_sub[j - sep_index] = j;
            }

            blaze::DynamicMatrix<double> Sigma_RdotS
                = inv_pos_def(submatrix_view_square(K_prime, R_sub));

            blaze::DynamicMatrix<double> Gamma_RdotS
                = -Sigma_RdotS * submatrix_view(K_prime, R_sub, sep_sub);

            auto Sigma_sep = submatrix_view_square(Sigma, seps[i]);
            auto tmp = Gamma_RdotS * Sigma_sep;
            submatrix_assign(Sigma, tmp, R, seps[i]);
            submatrix_assign(Sigma, trans(tmp), seps[i], R);

            blaze::DynamicMatrix<double>
                Sigma_sep_inv = inv_pos_def(Sigma_sep);

            blaze::DynamicMatrix<double>
                Sigma_R_sep = submatrix_view(Sigma, R, seps[i]),
                Sigma_R = declsym(
                    Sigma_RdotS + Sigma_R_sep*Sigma_sep_inv*trans(Sigma_R_sep)
                );

            submatrix_assign_square(Sigma, Sigma_R, R);
            
            // Equation 7 from
            // Carvalho et al. (2007, doi:10.1093/biomet/asm056)
            blaze::DynamicMatrix<double> Sigma_prime_inv
                = inv_pos_def(submatrix_view_square(Sigma, primes[i + 1]));

            submatrix_assign_square(K_c, Sigma_prime_inv, primes[i + 1], true);
            submatrix_assign_square(K_c, -Sigma_sep_inv, seps[i], true);
        }

        submatrix_assign_square(K, K_c, component);
        igraph_destroy(&G_c);  // Release memory.
    }

    return K;
}


blaze::DynamicMatrix<double> rgwish_identity(
    igraph_t* G_ptr, double df, sfc64& rng
) {
    int max_prime;
    return rgwish_identity(G_ptr, df, rng, max_prime);
}


blaze::DynamicMatrix<double> rgwish(
    igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng,
    int& max_prime
) {
    max_prime = 1;  // Number of nodes of the largest prime component
    int p = igraph_vcount(G_ptr);
    if (igraph_ecount(G_ptr) == 0) return rgwish_empty(p, df, rate, rng);
    if (is_complete(G_ptr)) return rwish(df, rate, rng);

    // Split the graph into its connected components.
    std::vector<std::vector<double> > components = get_components(G_ptr);
    blaze::DynamicMatrix<double> K(p, p, 0.0);
    boost::random::chi_squared_distribution<> rchisq(df);

    for (std::vector<double> component : components) {
        int p_c = component.size();

        if (p_c == 1) {
            int i = component[0];
            K(i, i) = rchisq(rng) / rate(i, i);
            continue;
        }

        igraph_t G_c = get_subgraph(G_ptr, component);
        blaze::DynamicMatrix<double> rate_c = submatrix(rate, component);

        if (is_complete(&G_c)) {
            if (p_c > max_prime) max_prime = p_c;
            blaze::DynamicMatrix<double> K_c = rwish(df, rate_c, rng);
            submatrix_assign_square(K, K_c, component);
            igraph_destroy(&G_c);  // Release memory.
            continue;
        }

        auto [primes, seps] = decompose_graph(&G_c);
        if (primes[0].size() > max_prime) max_prime = primes[0].size();

        if (seps.size() == 0) {  // `G_c` is a prime graph.
            blaze::DynamicMatrix<double> K_c = rgwish_L(&G_c, df, rate_c, rng);
            submatrix_assign_square(K, K_c, component);
            igraph_destroy(&G_c);  // Release memory.
            continue;
        }

        // Start of the generation of the G-Wishart with the first prime.
        igraph_t G_prime = get_subgraph(&G_c, primes[0]);
        
        blaze::DynamicMatrix<double> K_c(p_c, p_c, 0.0), Sigma(p_c, p_c, 0.0),
            K_prime, K_prime_inv, rate_prime = submatrix(rate_c, primes[0]);

        if (is_complete(&G_prime)) {
            K_prime = rwish(df, rate_prime, rng);
            K_prime_inv = inv_pos_def(K_prime);
        } else {
            K_prime_inv = rginvwish_L(&G_prime, df, rate_prime, rng);
            K_prime = inv_pos_def(K_prime_inv);
        }

        igraph_destroy(&G_prime);

        // Equation 7 of Carvalho et al. (2007, doi:10.1093/biomet/asm056)
        submatrix_assign_square(K_c, K_prime, primes[0]);
        submatrix_assign_square(Sigma, K_prime_inv, primes[0]);
        
        for (int i = 0; i < seps.size(); i++) {
            if (primes[i + 1].size() > max_prime)
                max_prime = primes[i + 1].size();

            rate_prime = submatrix(rate_c, primes[i + 1]);
            G_prime = get_subgraph(&G_c, primes[i + 1]);
            K_prime = rgwish_L(&G_prime, df, rate_prime, rng);
            igraph_destroy(&G_prime);

            // R = list(set(prime) - set(sep))
            // `sep_sub` and `R_sub` contain the same indices as `sep`
            // and `R` but then relative to the current prime subgraph.
            // We exploit that seps[i] and primes[i + 1] are sorted.
            std::vector<double> sep_sub(seps[i].size()),
                R(primes[i + 1].size() - sep_sub.size()), R_sub(R.size());

            int sep_index = 0;

            for (int j = 0; j < primes[i + 1].size(); j++) if (
                sep_index < seps[i].size()
                    and primes[i + 1][j] == seps[i][sep_index]
            ) {
                sep_sub[sep_index] = j;
                sep_index++;
            } else {
                R[j - sep_index] = primes[i + 1][j];
                R_sub[j - sep_index] = j;
            }

            blaze::DynamicMatrix<double> Sigma_RdotS
                = inv_pos_def(submatrix_view_square(K_prime, R_sub));

            blaze::DynamicMatrix<double> Gamma_RdotS
                = -Sigma_RdotS * submatrix_view(K_prime, R_sub, sep_sub);

            auto Sigma_sep = submatrix_view_square(Sigma, seps[i]);
            auto tmp = Gamma_RdotS * Sigma_sep;
            submatrix_assign(Sigma, tmp, R, seps[i]);
            submatrix_assign(Sigma, trans(tmp), seps[i], R);

            blaze::DynamicMatrix<double>
                Sigma_sep_inv = inv_pos_def(Sigma_sep);

            blaze::DynamicMatrix<double>
                Sigma_R_sep = submatrix_view(Sigma, R, seps[i]),
                Sigma_R = declsym(
                    Sigma_RdotS + Sigma_R_sep*Sigma_sep_inv*trans(Sigma_R_sep)
                );

            submatrix_assign_square(Sigma, Sigma_R, R);
            
            // Equation 7 from
            // Carvalho et al. (2007, doi:10.1093/biomet/asm056)
            blaze::DynamicMatrix<double> Sigma_prime_inv
                = inv_pos_def(submatrix_view_square(Sigma, primes[i + 1]));

            submatrix_assign_square(K_c, Sigma_prime_inv, primes[i + 1], true);
            submatrix_assign_square(K_c, -Sigma_sep_inv, seps[i], true);
        }

        submatrix_assign_square(K, K_c, component);
        igraph_destroy(&G_c);  // Release memory.
    }

    return K;
}


blaze::DynamicMatrix<double> rgwish(
    igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    int max_prime;
    return rgwish(G_ptr, df, rate, rng, max_prime);
}


double proposal_G_es(int p, int n_e_tilde, int n_e) {
    // Proposal transition probability from `G` to `G_tilde` based on edge
    // counts
    int max_e = p * (p - 1) / 2;

    if (n_e == 0 or n_e == max_e) {
        return 1.0 / max_e;
    } else if (n_e > n_e_tilde) {
        return 0.5 / n_e;
    } else {
        return 0.5 / (max_e - n_e);
    }
}


igraph_t adj2igraph(blaze::DynamicMatrix<int>& adj, int n_e) {
    int p = adj.rows();
    igraph_t G;
    igraph_empty(&G, p, false);
    igraph_vector_t edge_list;
    igraph_vector_init(&edge_list, 2 * n_e);
    int vec_ind = 0;

    for (int i = 0; i < p; i++) for (int j = i + 1; j < p; j++)
        if (adj(i, j)) {
            VECTOR(edge_list)[vec_ind++] = i;
            VECTOR(edge_list)[vec_ind++] = j;
        }

    igraph_add_edges(&G, &edge_list, 0);
    igraph_vector_destroy(&edge_list);
    return G;
}


template <typename T>
blaze::DynamicMatrix<T> permute_mat(
    blaze::DynamicMatrix<T>& mat, std::vector<int>& perm
) {
    std::vector<int> from(0), to(0);

    for (int i = 0; i < perm.size(); i++) if (perm[i] != i) {
        from.push_back(perm[i]);
        to.push_back(i);
    }

    blaze::DynamicMatrix<T> perm_rows = rows(mat, from);
    columns(perm_rows, to) = columns(perm_rows, from);
    blaze::DynamicMatrix<T> mat_perm(mat);
    rows(mat_perm, to) = perm_rows;
    columns(mat_perm, to) = trans(perm_rows);
    return mat_perm;
}


double log_N_tilde(
    blaze::LowerMatrix<blaze::DynamicMatrix<double> >& Phi,
    double rate_perm_11, double rate_perm_21
) {
    /*
    The log of the function N from
    Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746)

    `rate_perm_11` and `rate_perm_21` contain the element in the last row and
    column and the element just above it in the rate matrix, respectively.
    */
    int p = Phi.rows();

    return std::log(Phi(p - 2, p - 2)) + 0.5*(
        -std::log(rate_perm_11) + rate_perm_11*std::pow(
            -sum(
                submatrix(Phi, p - 2, 0, 1, p - 2)
                    % submatrix(Phi, p - 1, 0, 1, p - 2)
            // The plus in the next line is a minus in Cheng & Lengkoski
            // (2012). I believe that the minus is a typo in the article.
            )/Phi(p - 2, p - 2) + Phi(p - 2, p - 2)*rate_perm_21/rate_perm_11,
            2
        )
    );
}


double log_norm_ratio_Letac(
    blaze::DynamicMatrix<int>& adj, int i, int j, double df_0
) {
    /*
    Log of the approximation of the ratio of normalizing constants of the
    G-Wishart prior distributions from Letac et al. (2018, arXiv:1706.04416v2).

    The ratio is evaluated at the graph given by adjacency matrix `adj` with
    edge (`i`, `j`) absent (numerator) divided by the same graph with
    (`i`, `j`) present (denominator).

    `df_0` is the degrees of freedom of the G-Wishart prior distribution.
    */
    // `n_paths` is the number of paths of length 2 that connect nodes `i` and
    // `j`.
    int p = adj.rows(), n_paths = 0;
    for (int l = 0; l < p; l++) if (adj(i, l) and adj(j, l)) n_paths++;

    return std::log(0.5) - 0.5 * std::log(M_PI)
        + std::lgamma(0.5 * (df_0 + n_paths))
        - std::lgamma(0.5 * (df_0 + n_paths + 1.0));
}


std::tuple<std::vector<int>, std::vector<int>> permute_e_last(
    int i, int j, int p
) {
    /*
    Permute the nodes such that edge `(i, j)` becomes edges (p - 1, p).

    This function returns the permutation and inverse permutation.
    */
    std::vector<int> perm(p), perm_inv(p);
    std::iota(perm.begin(), perm.end(), 0);

    // Permute the nodes involved in `e`.
    if (i != p - 2) {
        perm[i] = p - 2;

        if (j == p - 2) {
            perm[p - 2] = p - 1;
            perm[p - 1] = i;
        } else {
            perm[p - 2] = i;
            perm[p - 1] = j;
            perm[j] = p - 1;
        }
    }

    for (int l = 0; l < p; l++) perm_inv[perm[l]] = l;    
    return std::make_tuple(perm, perm_inv);
}


double log_balancing_function(double log_t) {
    /*
    Compute the log of the balancing function t/(1+t).

    `log_t` is the log of t.

    This function equals -log1pexp(-`log_t`). We use Equation 10 from
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    to compute it.
    */
    if (log_t > 37.0) return -std::exp(-log_t);
    if (log_t > -18.0) return -std::log1p(std::exp(-log_t));
    if (log_t > -33.3) return log_t - std::exp(log_t);
    return log_t;
}


std::tuple<
    blaze::DynamicMatrix<double>, blaze::DynamicMatrix<double>, double
> locally_balanced_proposal(
    blaze::DynamicMatrix<double>& K, blaze::DynamicMatrix<int>& adj, int n_e,
    blaze::DynamicMatrix<double>& edge_prob_mat, double df_0,
    blaze::DynamicMatrix<double>& rate, bool Letac = true
) {
    /*
    Compute the locally balanced proposal from
    Zanella (2019, doi:10.1080/01621459.2019.1585255)
    */
    int p = adj.rows();
    blaze::DynamicMatrix<double> log_Q(p, p, -INFINITY);
    // The matrix `Phi` is specified here to avoid repeated expensive memory
    // (de)allocation.
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi;

    double log_q_add = std::log(proposal_G_es(p, n_e + 1, n_e)),
        log_q_rm = std::log(proposal_G_es(p, n_e - 1, n_e));

    // We measure the time to infer compute time in higher core compute
    // environments.
    std::chrono::time_point<std::chrono::steady_clock>
        start_time = std::chrono::steady_clock::now();

    // Fused triangular loop based on 
    // https://stackoverflow.com/a/33836073/5216563 as OnepMP does not support
    // collapsing triangular loops.
    #pragma omp parallel for schedule(static) private(Phi)
    for (int e_id = 0; e_id < p * (p - 1) / 2; e_id++) {
        int i = e_id / p, j = e_id % p;

        if (i >= j) {
            i = p - i - 2;
            j = p - j - 1;
        }

        auto [perm, perm_inv] = permute_e_last(i, j, p);
        blaze::DynamicMatrix<double> K_perm = permute_mat(K, perm_inv);
        llh(K_perm, Phi);
        int exponent;

        if (adj(i, j)) {
            exponent = -1;
            log_Q(i, j) = log_q_rm;
        } else {
            exponent = 1;
            log_Q(i, j) = log_q_add;
        }

        log_Q(i, j) += log_balancing_function(exponent * (
            std::log(edge_prob_mat(i, j)) - std::log1p(-edge_prob_mat(i, j))
                + log_N_tilde(
                    Phi, rate(perm_inv[p - 1], perm_inv[p - 1]),
                    rate(perm_inv[p - 2], perm_inv[p - 1])
                ) + Letac*log_norm_ratio_Letac(adj, i, j, df_0)
        ));
    }

    double par_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time
    ).count();

    log_Q -= max(log_Q);
    blaze::DynamicMatrix<double> Q = exp(log_Q);
    double sum_Q = sum(Q);
    return std::make_tuple(Q / sum_Q, log_Q - std::log(sum_Q), par_time);
}


void update_K_from_Phi(
    int p, int j, std::vector<int>& perm, blaze::DynamicMatrix<double>& K,
    blaze::LowerMatrix<blaze::DynamicMatrix<double> >& Phi
) {
    /*
    Update the precision matrix `K` in place according to a new Phi.

    The matrix `Phi` has been permuted such that indices `i` and `j` become
    indices p-1 and p. Matrix `K` is unpermuted.
    */
    blaze::DynamicVector<double> K_vec = Phi * trans(row(Phi, p - 1));
    K_vec = elements(K_vec, perm);  // Undo the permutation.
    column(K, j) = K_vec;
    row(K, j) = trans(K_vec);
}


double update_single_edge(
    blaze::DynamicMatrix<double>& K, blaze::DynamicMatrix<int>& adj, int& n_e,
    blaze::DynamicMatrix<double>& edge_prob_mat, double df, double df_0,
    blaze::DynamicMatrix<double>& rate, sfc64& rng,
    blaze::LowerMatrix<blaze::DynamicMatrix<double> >& Phi,
    bool approx = false, bool delayed_accept = true, bool loc_bal = true,
    bool Letac = true
) {
    /*
    MCMC step that attempts to update a single edge

    This function modifies `K`, `adj` and `n_e` in place.

    The MCMC is similar to the one described in
    Cheng & Lenkoski (2012, doi:10.1214/12-EJS746).

    The matrix `Phi` is passed to avoid potentially expensive memory
    (de)allocation.

    `approx` indicates whether the target is approximated through the
    approximation of the ratio of normalizing constants of the G-Wishart prior
    distributions from Letac et al. (2018, arXiv:1706.04416v2).

    `delayed_accept` indicates whether the delayed acceptance MCMC in
    Algorithm 1 of Christen & Fox (2005, doi:10.1198/106186005X76983) is used.
    The surrogate posterior considered derives from the approximation of the
    ratio of normalizing constants of the G-Wishart prior distributions from
    Letac et al. (2018, arXiv:1706.04416v2).

    `approx` and `delayed_accept` cannot be true simultaneously.

    `loc_bal` indicates whether to use the locally balanced proposal from
    Zanella (2019, doi:10.1080/01621459.2019.1585255).

    `Letac` indicates whether to use the approximation for the ratio of
    normalization constants from Letac et al. (2018, arXiv:1706.04416v2) or to
    instead approximate the ratio by one.
    */
    if (approx and delayed_accept) throw std::runtime_error(
        "`approx` and `delayed_accept` cannot be true simultaneously."
    );

    // (i, j) is the edge to be updated.
    int i, j, p = adj.rows(), max_e = p * (p - 1) / 2;
    bool accept = true, add;  // Whether an edge is being added
    // The steps and notation follow Algorithm 1 of Christen & Fox (2005).
    boost::random::uniform_01<double> runif;

    double log_q_y_x, log_q_x_y,  // Proposal transition probabilities
        par_time = 0.0;  // Time spent on parallel computing.
    
    // Sample from the proposal
    if (loc_bal) {
        // Compute the locally balanced proposal.
        auto [Q, log_Q, tmp] = locally_balanced_proposal(
            K, adj, n_e, edge_prob_mat, df_0, rate, Letac
        );

        par_time += tmp;

        // Sample an edge from the proposal using the inverse CDF method.
        i = 0;
        j = 0;
        double tmp_sum = 0.0, U = runif(rng);

        do {
            if (j == p - 1) j = ++i;
            tmp_sum += Q(i, ++j);
        } while (tmp_sum < U);

        add = not adj(i, j);
        log_q_y_x = log_Q(i, j);
    } else {
        // Decide whether to propose an edge addition or removal.
        if (n_e == 0) {
            add = true;
        } else if (n_e == max_e) {
            add = false;
        } else {
            add = runif(rng) < 0.5;
        }

        // Pick edge to add or remove uniformly at random.
        int row_sum, row_sum_cum = 0;
        i = -1;

        if (add) {
            boost::random::uniform_int_distribution<int>
                r_edge(1, max_e - n_e);

            int e_id = r_edge(rng);

            do {
                i++;

                row_sum
                    = p - i - 1 - sum(submatrix(adj, i, i + 1, 1, p - i - 1));
                
                row_sum_cum += row_sum;
            } while (row_sum_cum < e_id);

            e_id -= row_sum_cum - row_sum;
            j = i;
            row_sum = 0;
            while (row_sum < e_id) row_sum += 1 - adj(i, ++j);
        } else {
            boost::random::uniform_int_distribution<int> r_edge(1, n_e);
            int e_id = r_edge(rng);

            do {
                i++;
                row_sum = sum(submatrix(adj, i, i + 1, 1, p - i - 1));
                row_sum_cum += row_sum;
            } while (row_sum_cum < e_id);

            e_id -= row_sum_cum - row_sum;
            j = i;
            row_sum = 0;
            while (row_sum < e_id) row_sum += adj(i, ++j);
        }

        int n_e_tilde = n_e + 2*add - 1;
        log_q_y_x = std::log(proposal_G_es(p, n_e_tilde, n_e));
        log_q_x_y = std::log(proposal_G_es(p, n_e, n_e_tilde));
    }

    int exponent = 2*add - 1, n_e_tilde = n_e + exponent;
    auto [perm, perm_inv] = permute_e_last(i, j, p);
    blaze::DynamicMatrix<double> K_perm = permute_mat(K, perm_inv);
    llh(K_perm, Phi);

    double log_target_ratio_approx, log_g_x_y,
        rate_pp = rate(perm_inv[p - 1], perm_inv[p - 1]),
        rate_1p = rate(perm_inv[p - 2], perm_inv[p - 1]),
        log_N_tilde_post = log_N_tilde(Phi, rate_pp, rate_1p),
        log_prior_ratio = exponent * (
            std::log(edge_prob_mat(i, j)) - std::log1p(-edge_prob_mat(i, j))
        );

    double Phi_12_cur = Phi(p - 1, p - 2);
    boost::random::chi_squared_distribution<> rchisq(df);
    Phi(p - 1, p - 1) = std::sqrt(rchisq(rng) / rate_pp);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);

    if (loc_bal) {
        // Compute the reverse transition probability `log_q_x_y`.

        // Update `Phi(p - 1, p - 2)` according to the proposal.
        if (add) {  // The graph contains (`i`, `j`) in the proposal.
            Phi(p - 1, p - 2) = rnorm(rng)/std::sqrt(rate_pp)
                - Phi(p - 2, p - 2)*rate_1p/rate_pp;
        } else {  // The graph does not contain (`i`, `j`) in the proposal.
            Phi(p - 1, p - 2) = -sum(
                submatrix(Phi, p - 2, 0, 1, p - 2)
                    % submatrix(Phi, p - 1, 0, 1, p - 2)
            ) / Phi(p - 2, p - 2);
        }

        // Make `adj` equal to the adjacency matrix of the proposed graph.
        adj(i, j) = add;
        adj(j, i) = add;
        update_K_from_Phi(p, j, perm, K, Phi);

        auto [Q_tilde, log_Q_tilde, tmp] = locally_balanced_proposal(
            K, adj, n_e_tilde, edge_prob_mat, df_0, rate, Letac
        );

        par_time += tmp;
        log_q_x_y = log_Q_tilde(i, j);
    }

    if (delayed_accept or approx) {
        log_target_ratio_approx = log_prior_ratio + exponent*(
            log_N_tilde_post + Letac*log_norm_ratio_Letac(adj, i, j, df_0)
        );

        log_g_x_y
            = std::min(0.0, log_q_x_y - log_q_y_x + log_target_ratio_approx);

        // Step 2
        accept = std::log(runif(rng)) < log_g_x_y;
    }

    if (accept and not approx) {
        double log_q_star_y_x ,log_q_star_x_y;

        if (delayed_accept) {
            log_q_star_y_x = log_g_x_y + log_q_y_x;

            log_q_star_x_y
                = std::min(log_q_x_y, log_q_y_x - log_target_ratio_approx);
        } else {
            log_q_star_y_x = log_q_y_x;
            log_q_star_x_y = log_q_x_y;
        }

        // Step 3
        // Exchange algorithm to avoid evaluation of normalization constants
        blaze::DynamicMatrix<int> adj_perm = permute_mat(adj, perm_inv);
        adj_perm(p - 2, p - 1) = add;
        adj_perm(p - 1, p - 2) = add;
        igraph_t G_perm = adj2igraph(adj_perm, n_e_tilde);

        blaze::DynamicMatrix<double>
            K_0_tilde = rgwish_identity(&G_perm, df_0, rng);

        igraph_destroy(&G_perm);
        blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi_0_tilde;
        llh(K_0_tilde, Phi_0_tilde);

        // The log of the function N from
        // Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746) with
        // identity rate matrix
        double log_N_tilde_prior
            = std::log(Phi_0_tilde(p - 2, p - 2)) + 0.5*std::pow(
                sum(
                    submatrix(Phi_0_tilde, p - 2, 0, 1, p - 2)
                        % submatrix(Phi_0_tilde, p - 1, 0, 1, p - 2)
                ) / Phi_0_tilde(p - 2, p - 2),
                2
            );

        double log_target_ratio = log_prior_ratio + exponent*(
            log_N_tilde_post - log_N_tilde_prior
        );

        accept = std::log(runif(rng))
            < log_q_star_x_y - log_q_star_y_x + log_target_ratio;
    }

    if (accept) {  // Update the graph.
        adj(i, j) = add;
        adj(j, i) = add;
        n_e = n_e_tilde;
    } else if (loc_bal) {  // Revert any update in `adj` and `Phi`.
        adj(i, j) = not add;
        adj(j, i) = not add;
        Phi(p - 1, p - 2) = Phi_12_cur;
    }

    if (not loc_bal) {
        // Update `Phi(p - 1, p - 2)`.
        if (adj(i, j)) {  // The graph contains (`i`, `j`) after updating.
            Phi(p - 1, p - 2) = rnorm(rng)/std::sqrt(rate_pp)
                - Phi(p - 2, p - 2)*rate_1p/rate_pp;
        } else {  // The graph does not contain (`i`, `j`) after updating.
            Phi(p - 1, p - 2) = -sum(
                submatrix(Phi, p - 2, 0, 1, p - 2)
                    % submatrix(Phi, p - 1, 0, 1, p - 2)
            ) / Phi(p - 2, p - 2);
        }
    }

    update_K_from_Phi(p, j, perm, K, Phi);  // Not required if `loc_bal and accept`?
    return par_time;
}


double update_G_cpp(
    int p, long* adj_in, double* edge_prob_mat_in, double df, double df_0,
    double* rate_in, int n_edge, long seed, bool approx = false,
    bool delayed_accept = true, bool loc_bal = true, bool Letac = true,
    bool get_log_lik = false
) {
    /*
    `p` is the number of nodes.
    `adj_in` is the adjacency matrix of the graph.
    `n_edge` is the number of single edge MCMC steps that are attempted.

    `adj_in` is modified in place.

    Updating an `igraph_t` one edge at a time comes with a notable
    computational cost if `approx = true`. We therefore work with the
    adjacency matrix.
    */
    sfc64 rng(seed);
    blaze::DynamicMatrix<int> adj(p, p, adj_in);

    blaze::DynamicMatrix<double> rate(p, p, rate_in),
        edge_prob_mat(p, p, edge_prob_mat_in);

    int n_e = sum(adj) / 2;
    igraph_t G = adj2igraph(adj, n_e);
    blaze::DynamicMatrix<double> K = rgwish(&G, df, rate, rng);
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi(p);
    // Time spent on parallel computing or the liglikelihood
    double par_time_or_log_lik = 0.0;

    for (int i = 0; i < n_edge; i++) par_time_or_log_lik += update_single_edge(
        K, adj, n_e, edge_prob_mat, df, df_0, rate, rng, Phi, approx,
        delayed_accept, loc_bal, Letac
    );

    // Copy `adj` to `adj_in`.
    for (int i = 0; i < p; i++)  {
        int ixp = i * p;
        for (int j = 0; j < p; j++) adj_in[ixp + j] = adj(i, j);
    }
    
    // Compute the log-likelihood
    // 0.5*n*(logdet(K) - p*log(2 * pi)) - trace(K * U).
    if (get_log_lik) {
        par_time_or_log_lik
            = 0.5*(df - df_0)*(std::log(det(K)) - p*std::log(2 * M_PI));

        blaze::DynamicMatrix<double> U = rate;

        for (int i = 0; i < p; i++) {
            U(i, i) -= 1.0;  // U = rate - I_p
            par_time_or_log_lik -= inner(row(U, i), row(K, i));
        }
    }
    
    return par_time_or_log_lik;
}


void update_G_DCBF_cpp(
    int p, long* adj_in, double* edge_prob_mat_in, double df, double df_0,
    double* rate_in, long seed
) {
    /*
    This impelements the direct double conditional Bayes factor sampler from
    Hinne et al. (2014, doi:10.1002/sta4.66).

    `p` is the number of nodes.
    `adj_in` is the adjacency matrix of the graph.

    `adj_in` is modified in place.
    */
    sfc64 rng(seed);
    blaze::DynamicMatrix<int> adj(p, p, adj_in);

    blaze::DynamicMatrix<double> rate(p, p, rate_in),
        edge_prob_mat(p, p, edge_prob_mat_in);

    int n_e = sum(adj) / 2;
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi(p);
    boost::random::uniform_01<double> runif;

    for (int rep = 0; rep < p; rep++) {
        // Sample uniformly which edge (i, j) to update.
        boost::random::uniform_int_distribution<int>
            r_edge(1, p * (p - 1) / 2);

        int i = 0, j = r_edge(rng);
        while (j > p - i - 1) j -= p - ++i;
        j += i;
        igraph_t G = adj2igraph(adj, n_e);
        blaze::DynamicMatrix<double> K = rgwish(&G, df, rate, rng);
        igraph_destroy(&G);

        // (i, j) is the edge to be updated.
        bool add = not adj(i, j);  // Whether an edge is being added

        int exponent = 2*add - 1, n_e_tilde = n_e + exponent;
        auto [perm, perm_inv] = permute_e_last(i, j, p);
        blaze::DynamicMatrix<double> K_perm = permute_mat(K, perm_inv);
        llh(K_perm, Phi);

        double log_prior_ratio = exponent * (
                std::log(edge_prob_mat(i, j))
                    - std::log1p(-edge_prob_mat(i, j))
            ),
            log_N_tilde_post = log_N_tilde(
                Phi, rate(perm_inv[p - 1], perm_inv[p - 1]),
                rate(perm_inv[p - 2], perm_inv[p - 1])
            );

        // Step 3
        // Exchange algorithm to avoid evaluation of normalization constants
        blaze::DynamicMatrix<int> adj_perm = permute_mat(adj, perm_inv);
        adj_perm(p - 2, p - 1) = add;
        adj_perm(p - 1, p - 2) = add;
        igraph_t G_perm = adj2igraph(adj_perm, n_e_tilde);

        blaze::DynamicMatrix<double>
            K_0_tilde = rgwish_identity(&G_perm, df_0, rng);

        igraph_destroy(&G_perm);
        blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi_0_tilde;
        llh(K_0_tilde, Phi_0_tilde);

        // The log of the function N from
        // Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746) with
        // identity rate matrix
        double log_N_tilde_prior
            = std::log(Phi_0_tilde(p - 2, p - 2)) + 0.5*std::pow(
                sum(
                    submatrix(Phi_0_tilde, p - 2, 0, 1, p - 2)
                        % submatrix(Phi_0_tilde, p - 1, 0, 1, p - 2)
                ) / Phi_0_tilde(p - 2, p - 2),
                2
            );

        double log_target_ratio = log_prior_ratio + exponent*(
            log_N_tilde_post - log_N_tilde_prior
        );

        if (std::log(runif(rng)) < log_target_ratio) {  // Update the graph.
            adj(i, j) = add;
            adj(j, i) = add;
            n_e = n_e_tilde;
        }
    }

    // Copy `adj` to `adj_in`.
    for (int i = 0; i < p; i++)  {
        int ixp = i * p;
        for (int j = 0; j < p; j++) adj_in[ixp + j] = adj(i, j);
    }
}


void update_K(
    blaze::DynamicMatrix<double>& K, igraph_t& G, double df,
    blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    /*
    Update K using the maximum clique block Gibbs sampler
    (Wang & Li, 2012, Section 2.4, doi:10.1214/12-EJS669).
    */
    int p = igraph_vcount(&G);
    igraph_vector_ptr_t igraph_cliques;
    igraph_vector_ptr_init(&igraph_cliques, 0);
    igraph_maximal_cliques(&G, &igraph_cliques, 0, 0);

    for (int i = 0; i < igraph_vector_ptr_size(&igraph_cliques); i++) {
        igraph_vector_t* clique_ptr
            = (igraph_vector_t*) igraph_vector_ptr_e(&igraph_cliques, i);

        std::vector<int> clique(igraph_vector_size(clique_ptr)),
            not_clique(p - clique.size());

        for (int j = 0; j < clique.size(); j++)
            clique[j] = igraph_vector_e(clique_ptr, j);

        std::sort(clique.begin(), clique.end());
        int tmp_ind = 0;

        for (int j = 0; j < p; j++) {
            if (tmp_ind < clique.size() and j == clique[tmp_ind]) {
                tmp_ind++;
            } else {
                not_clique[j - tmp_ind] = j;
            }
        }

        blaze::DynamicMatrix<double> chol_inv_B,
            rate_clique = submatrix_view_square(rate, clique),
            K_not_clique = submatrix_view_square(K, not_clique),
            A = rwish(df, rate_clique, rng),
            B = submatrix_view(K, not_clique, clique);

        blaze::LowerMatrix<blaze::DynamicMatrix<double> > chol;
        llh(K_not_clique, chol);
        solve(chol, chol_inv_B, B);

        submatrix_assign_square(
            K, A + declsym(trans(chol_inv_B) * chol_inv_B), clique
        );
    }

    IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR(
        &igraph_cliques, igraph_vector_destroy
    );

    igraph_vector_ptr_destroy_all(&igraph_cliques);
}


void update_G_CL_cpp(
    int p, long* adj_in, double* K_in, double* edge_prob_mat_in, double df,
    double df_0, double* rate_in, long seed
) {
    /*
    This impelements the CL algorithm from
    Cheng & Lengkoski (2012, Section 2.4, doi:10.1214/12-EJS746).

    `p` is the number of nodes.
    `adj_in` is the adjacency matrix of the graph.
    `K_in` is the precision matrix.

    `adj_in` and `K_in` are modified in place.
    */
    sfc64 rng(seed);
    blaze::DynamicMatrix<int> adj(p, p, adj_in);

    blaze::DynamicMatrix<double> K(p, p, K_in), rate(p, p, rate_in),
        rate_0(p, p, 0.0), edge_prob_mat(p, p, edge_prob_mat_in);

    for (int i = 0; i < p; i++) rate_0(i, i) = 1.0;
    int n_e = sum(adj) / 2;
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi(p);
    boost::random::uniform_01<double> runif;
    boost::random::chi_squared_distribution<> rchisq(df);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);

    for (int rep = 0; rep < p; rep++) {
        // Sample uniformly which edge (i, j) to update.
        boost::random::uniform_int_distribution<int>
            r_edge(1, p * (p - 1) / 2);

        int i = 0, j = r_edge(rng);
        while (j > p - i - 1) j -= p - ++i;
        j += i;
        bool add = not adj(i, j);  // Whether an edge is being added
        int exponent = 2*add - 1, n_e_tilde = n_e + exponent;
        auto [perm, perm_inv] = permute_e_last(i, j, p);
        blaze::DynamicMatrix<double> K_perm = permute_mat(K, perm_inv);
        llh(K_perm, Phi);

        double rate_pp = rate(perm_inv[p - 1], perm_inv[p - 1]),
            rate_1p = rate(perm_inv[p - 2], perm_inv[p - 1]),
            log_prior_ratio = exponent * (
                std::log(edge_prob_mat(i, j))
                    - std::log1p(-edge_prob_mat(i, j))
            ),
            log_N_tilde_post = log_N_tilde(Phi, rate_pp, rate_1p);

        // Decide whether to promote `G_tilde`
        if (
            std::log(runif(rng)) < log_prior_ratio + exponent*log_N_tilde_post
        ) {
            // Double Metropolis-Hastings step to avoid evaluation of
            // normalization constants
            blaze::DynamicMatrix<int> adj_perm = permute_mat(adj, perm_inv);
            adj_perm(p - 2, p - 1) = add;
            adj_perm(p - 1, p - 2) = add;
            blaze::DynamicMatrix<double> K_0_tilde(K_perm);
            igraph_t G_perm = adj2igraph(adj_perm, n_e_tilde);
            update_K(K_0_tilde, G_perm, df_0, rate_0, rng);
            igraph_destroy(&G_perm);
            blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi_0_tilde;
            llh(K_0_tilde, Phi_0_tilde);

            // The log of the function N from
            // Cheng & Lengkoski (2012, page 2314, doi:10.1214/12-EJS746) with
            // identity rate matrix
            double log_N_tilde_prior
                = std::log(Phi_0_tilde(p - 2, p - 2)) + 0.5*std::pow(
                    sum(
                        submatrix(Phi_0_tilde, p - 2, 0, 1, p - 2)
                            % submatrix(Phi_0_tilde, p - 1, 0, 1, p - 2)
                    ) / Phi_0_tilde(p - 2, p - 2),
                    2
                );

            double log_target_ratio = log_prior_ratio + exponent*(
                log_N_tilde_post - log_N_tilde_prior
            );

            if (std::log(runif(rng)) < -exponent * log_N_tilde_prior) {
                // Update the graph.
                adj(i, j) = add;
                adj(j, i) = add;
                n_e = n_e_tilde;
            }
        }

        // Update `Phi(p - 1, p - 2)` and `Phi(p - 1, p - 1)` according to the
        // current graph.
        if (adj(i, j)) {  // The graph contains (`i`, `j`).
            Phi(p - 1, p - 2) = rnorm(rng)/std::sqrt(rate_pp)
                - Phi(p - 2, p - 2)*rate_1p/rate_pp;
        } else {  // The graph does not contain (`i`, `j`)
            Phi(p - 1, p - 2) = -sum(
                submatrix(Phi, p - 2, 0, 1, p - 2)
                    % submatrix(Phi, p - 1, 0, 1, p - 2)
            ) / Phi(p - 2, p - 2);
        }

        Phi(p - 1, p - 1) = std::sqrt(rchisq(rng) / rate_pp);
        update_K_from_Phi(p, j, perm, K, Phi);
    }

    // Update K
    igraph_t G = adj2igraph(adj, n_e);
    update_K(K, G, df, rate, rng);
    igraph_destroy(&G);

    // Copy `adj` to `adj_in` and `K` to `K_in`.
    for (int i = 0; i < p; i++)  {
        int ixp = i * p;

        for (int j = 0; j < p; j++) {
            adj_in[ixp + j] = adj(i, j);
            K_in[ixp + j] = K(i, j);
        }
    }
}


void rgwish_cpp(
    double* K_out, igraph_t* G_ptr, double df, double* rate_in, long seed,
    int& max_prime, bool decompose = true
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G_ptr);
    blaze::DynamicMatrix<double> rate(p, p, rate_in);

    blaze::DynamicMatrix<double> K = decompose
        ? rgwish(G_ptr, df, rate, rng, max_prime)
        : rgwish_L(G_ptr, df, rate, rng);

    // Copy `K` to `K_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K_out[ixp + j] = K(i, j);
    }
}


void rgwish_identity_cpp(
    double* K_out, igraph_t* G_ptr, double df, long seed, int& max_prime,
    bool decompose = true
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G_ptr);

    blaze::DynamicMatrix<double> K = decompose
        ? rgwish_identity(G_ptr, df, rng, max_prime)
        : rgwish_L_identity(G_ptr, df, rng);

    // Copy `K` to `K_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K_out[ixp + j] = K(i, j);
    }
}


int main() {
    return 0;
}