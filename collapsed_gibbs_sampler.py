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
    theta = random_state.dirichlet(alpha)
    m = random_state.multinomial(N, theta)
    z = np.repeat(np.arange(K), m)
    random_state.shuffle(z)
    return z

def sample_mu(z, X, K, D, N, sigma, zeta, random_state):
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

# Load sample `X` and proceed with example parameters. (For the purpose of these project, these
# parameters should be identical to those used to generate the sample.)
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
    mu_lines.extend(ax.plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1, color=p['color']))

    x, y = X[z_ == k].T
    X_lines.extend(ax.plot(x, y, '.', color=p['color']))

# ... and sample forever.
for n in count(1):
    sys.stdout.write('\rSamples: %d ... ' % n)
    sys.stdout.flush()

    sleep(0.02)

    mu_, z_ = next(sampler)
    for k, ((x, y), mu_line, X_line) in enumerate(zip(mu_, mu_lines, X_lines)):
        mu_line.set_xdata(x)
        mu_line.set_ydata(y)

        x, y = X[z_ == k].T
        X_line.set_xdata(x)
        X_line.set_ydata(y)

    fig.canvas.draw()

    # This call helps the plot update continuously on some systems. See:
    # http://stackoverflow.com/a/19119738/3561
    plt.pause(0.1)
