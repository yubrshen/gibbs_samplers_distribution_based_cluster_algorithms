##########################################
# File: collapsed_gibbs_sampler.py       #
# Copyright Primer Technologies Inc 2017 #
##########################################

"""Usage: python collapsed_gibbs_sampler.py [input_filename]"""

import sys
from itertools import count, cycle
from numbers import Integral
from time import sleep

import numpy as np
from matplotlib import pyplot as plt


def initial_z(alpha, N, K, random_state):
    """
    Initialize z (component assignments)
    required to boostrap the gibbs sampling.
    alpha: the seed vector for Dirchlet sampling of the component proportions.
    N: the number of data points to which
    the sampling of component means and component assignments are performed.
    K: the assumed number of components in the data to be sampled.
    random_state: the random state used as consistent context
    for random sampling.

    It returns the computed z.
    """
    theta = random_state.dirichlet(alpha)
    m = random_state.multinomial(N, theta)
    z = np.repeat(np.arange(K), m)
    random_state.shuffle(z)
    return z

def sample_mu(z, X, K, D, N, sigma, zeta, random_state):
    """
    Perform the sampling of mu (the component means) in gibbs sample procedure.
    z: the component assignments prior or computed
    in the previous sampling iteration.
    X: the dataset of the data points to be sampled for
    component means and component assignments.
    K: the assumed number of components in the data to be sampled.
    D: the number of attributes of a data points in X.
    N: the number of data points in X to which
    the sampling of component means and component assignments are performed.
    sigma: the standard deviation of the noise to the data points
    zeta: the standard deviation for the Gaussian prior
    over the component means
    random_state: the random state used as consistent context
    for random sampling.

    It returns both the updated mu as numpy array and count_by_component,
    a list of counts of components in X according to z.

    """
    count_by_component = []
    mu = []
    for k in range(K):
        X_by_component_k = X[z == k]
        count_by_component_k = len(X_by_component_k)
        count_by_component.append(count_by_component_k)
        sum_by_component_k = (sum(X_by_component_k)
                              if (0 < len(X_by_component_k))
                              else np.zeros(D))
        denominator = count_by_component_k + sigma*sigma/(zeta*zeta)
        mean_for_mu_k = sum_by_component_k/denominator
        var_for_mu_k = (sigma*sigma/denominator)*np.eye(D)
        mu_k = random_state.multivariate_normal(mean_for_mu_k,
                                                var_for_mu_k, size=1)[0]
        mu.append(mu_k)
    return np.array(mu), count_by_component

def sample_z_collapsed(z, X, K, D, N, alpha, sigma, zeta, mu, random_state):
    """
    Perform the sampling of z (the component assignments)
    in collapsed gibbs sample procedure.
    z: the component assignments from the previous sampling iteration.
    X: the dataset of the data points to be sampled for
    component means and component assignments.
    K: the assumed number of components in the data to be sampled.
    D: the number of attributes of a data points in X.
    N: the number of data points in X to which
    the sampling of component means and component assignments are performed.
    alpha: the seed vector for Dirchlet sampling of the component proportions.
    sigma: the standard deviation of the noise to the data points
    zeta: the standard deviation for the Gaussian prior
    over the component means.
    mu: the component means.
    random_state: the random state used as consistent context
    for random sampling.

    It returns the updated z as numpy array.
    """

    diff_n_k_squared = np.array([y*y for y in [X-mu[k] for k in range(K)]])
    diff_n_k_sum = np.sum(diff_n_k_squared, axis=2)
    p_xn_k = np.exp(-diff_n_k_sum/(2.0*sigma))/np.power(2.0*np.pi*sigma, D/2.0)
    m_k_exclude_n = np.array([[len(X[[i for i in range(N) if (i != n and i == z[k])]])
                               for n in range(N)]
                              for k in range(K)])
    w = alpha[:, None] + m_k_exclude_n
    p_tild_xn_k = p_xn_k*w
    p_tild_xn_k_sum_over_k = np.sum(p_tild_xn_k, axis=0)
    p_tild_xn_k_sum_over_k[p_tild_xn_k_sum_over_k == 0] = 1.0
    # no normalization when the denominator is 0.0 to avoid div by 0.0 error
    # If the modeling is correct, the case should not happen
    p_z_n_k = p_tild_xn_k/p_tild_xn_k_sum_over_k
    z_next = []
    for n in range(N):
        try:
            z_next.append(random_state.choice(K, p=p_z_n_k[:,n]))
        except ValueError as ex:
            z_next.append(random_state.choice(K, p=None))
            print("Exception: {}; error probabilities: {}; \
            fix by random uniform".format(ex, p_z_n_k[:,n]))
        else:  # successful case
            pass
            # print("Indeed correct probabilities: {}".format(p_z_n_k[:,n]))

    return np.array(z_next)

def collapsed_gibbs_sampler(X, alpha, zeta, sigma, random_state=None):
    """A collapsed (serial) Gibbs sampler for a Gaussian Mixture Model with known
      `alpha`,
      `zeta`
      and
      `sigma`."""
    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)

    X = np.atleast_2d(X)
    N, D = X.shape

    alpha = np.atleast_1d(alpha)
    K, = alpha.shape
    assert (alpha > 0.0).all()

    assert zeta > 0.0
    assert sigma > 0.0

    # Initialize `z`.
    z = initial_z(alpha, N, K, random_state)
    # z = [np.random.choice(K) for n in range(N)]

    # Allocate `mu`. (This does *not* initialize it.)
    # mu = np.empty((K, D)) # no longer needed

    while True:
        # Sample each `mu`.
        mu, _ = sample_mu(z, X, K, D, N, sigma, zeta, random_state)

        # Sample each `z`.
        z = sample_z_collapsed(z, X, K, D, N, alpha, sigma, zeta,
                               mu, random_state)

        yield mu, z

def main():
    # Load sample `X` and proceed with example parameters.
    # (For the purpose of these project, these parameters should be identical
    # to those used to generate the sample.)
    
    X = np.loadtxt(sys.argv[1] if len(sys.argv) > 1 else 'X.tsv')
    
    alpha = [1.0, 1.0, 1.0]
    zeta = 2.0
    sigma = 1.0
    print 'alpha:', alpha
    print 'zeta:', zeta
    print 'sigma:', sigma

    # Initialize `sampler` and take a single sample
    # (required to initialize the visualization).
    sampler = collapsed_gibbs_sampler(X, alpha, zeta, sigma, random_state=0)

    original = [[0, 1],
                [1, 1],
                [1, 0]]
    
    sampled = [[0.9, 0.7],
               [0.3, 0.5],
               [0.6, 0.1]]
    
    # from the indices of sampled to those of the original
    expected_map = {0:1,
                    1:0,
                    2:2}
    
    def index_alignment(sampled, original):
        """
        map the indices of the sampled to those of the original
        so that the data points of the mapped pair are the closest in distance.
    
        sampled: a list of data points sampled.
        original: a list of data points original
    
        The length of sampled and original should be equal.
    
        return a map of index from that of the sampled to that of the original
        """
        assert(len(sampled) == len(original))
        len_sampled = len(sampled)
        len_original = len(original)
        sampled = np.array(sampled)
        original = np.array(original)
        available = np.full(len_original, True)
        # indices whether the data point has been mathched
    
        result = {}
        mu_distances = []
        for i in range(0, len_sampled):
            least = -1
            for j in range(0, len_original):
                if available[j]:     # not yet matched
                    diff = sampled[i]-original[j]
                    inner_diff = np.inner(diff, diff)
                    if (least < 0) or (inner_diff < least):
                        least = inner_diff
                        idx_least = j
                    # end of if (least < 0) or (inner_diff < least)
                # end of if available[j] == 0
            # end of for j in range(idx_original, len_original)
            result[i] = idx_least
            mu_distances.append(np.sqrt(least))
            available[idx_least] = False
        # end of for i in range(idx_sampled, len_sampled)
        # reorder the distance from the perspective of the original:
        mu_distances_reordered = [None] * len(original)
        for i in range(len(original)):
            mu_distances_reordered[result[i]] = mu_distances[i]
        return result, mu_distances_reordered
    
    
    assert(index_alignment(sampled, original)[0] == expected_map)
    
    def z_errors_f(K, z_aligned, z_original):
        z_errors = np.zeros(K)
        for i in range(len(z_aligned)):
            if z_original[i] != z_aligned[i]:
                z_errors[int(z_original[i])] += 1
        z_original_component_counts \
            = map(len, [z_original[z_original == k] for k in range(K)])
        return (z_errors/np.array([x if x != 0 else 1
                              for x in z_original_component_counts]),
           sum(z_errors)/len(z_aligned))
    
    def errors_f(mu, mu_original, z, z_original):
        index_map, mu_distances = index_alignment(mu, mu_original)
    
        z_aligned = map(lambda i: index_map[i], z)
    
        z_errors, z_error_total = z_errors_f(len(original), z_aligned, z_original)
        return mu_distances, sum(mu_distances), z_errors, z_error_total
    
    mu_original = np.loadtxt('./mu.tsv')
    z_original = np.loadtxt('./z.tsv')
    mu_dist_acc = np.empty(shape=[3, 0])
    mu_dist_total_acc = []
    z_err_acc = np.empty(shape=[3, 0])
    z_err_total_acc = []
    
    mu_, z_ = next(sampler)
    
    # Setup visualization ...
    plt.ion()
    
    fig, ax = plt.subplots()
    
    ax.grid(True)
    ax.set_aspect('equal')
    
    prop_cycle = cycle(plt.rcParams['axes.prop_cycle'])
    
    mu_lines, X_lines = [], []
    for k, (x, y) in enumerate(mu_):
        p = next(prop_cycle)
        mu_lines.extend(ax.plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1,
                                color=p['color'], marker=(3+k, 0, 0)))
    
        x, y = X[z_ == k].T
        X_lines.extend(ax.plot(x, y, '.', color=p['color']))
    
    try:
        # ... and sample forever.
        for n in count(1):
            sys.stdout.write('\rSamples: %d ... ' % n)
            sys.stdout.flush()
    
            sleep(0.02)
    
            mu_, z_ = next(sampler)
    
            mu_distances, sum_mu_distances, z_errors, z_error_total \
                = errors_f(mu_, mu_original, z_, z_original)
            
            mu_dist_acc = np.append(mu_dist_acc,
                                    np.reshape(mu_distances, [3, 1]), axis=1)
            mu_dist_total_acc.append(sum_mu_distances)
            z_err_acc = np.append(z_err_acc,
                                  np.reshape(z_errors, [3, 1]), axis=1)
            z_err_total_acc.append(z_error_total)
    
            for k, ((x, y), mu_line, X_line) in enumerate(
                    zip(mu_, mu_lines, X_lines)):
                mu_line.set_xdata(x)
                mu_line.set_ydata(y)
    
                x, y = X[z_ == k].T
                X_line.set_xdata(x)
                X_line.set_ydata(y)
    
            fig.canvas.draw()
    
            # This call helps the plot update continuously on some systems. See:
            # http://stackoverflow.com/a/19119738/3561
            plt.pause(0.1)
    except KeyboardInterrupt as ex:
        plt.subplot(2, 1, 1)
        for i in range(mu_dist_acc.shape[0]):
            plt.plot(mu_dist_acc[i], marker=(3+i, 0, 0))
        plt.plot(mu_dist_total_acc)
        plt.ylabel('Error Distances of Means')
        
        plt.subplot(2, 1, 2)
        for i in range(z_err_acc.shape[0]):
            plt.plot(z_err_acc[i], marker=(3+i, 0, 0))
        plt.plot(z_err_total_acc)
        plt.ylabel('Error Component Assignments')
        plt.show()
        plt.pause(100)
        raise(ex)
    finally:
        pass

if __name__ == '__main__':
    main()
