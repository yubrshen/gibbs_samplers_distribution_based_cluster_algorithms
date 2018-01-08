##########################################
# File: sample_gmm.py                    #
# Copyright Primer Technologies Inc 2017.#
##########################################

"""Usage: python sample_gmm.py [random_state]"""

import sys
from itertools import cycle
from numbers import Integral

import numpy as np
from matplotlib import pyplot as plt


def sample_gmm(alpha, zeta, sigma, D, N, random_state=None):
    """Sample `N` data points from a `D` dimensional Gaussian Mixture Model. `alpha` is the
    parameter for the Dirichlet prior over the component proportions, `zeta` is the standard
    deviation of the isotropic Gaussian prior over the component means, and `sigma` is the standard
    deviation of the noise for each data point."""
    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)

    # `K` is the number of components.
    alpha = np.atleast_1d(alpha)
    K, = alpha.shape
    assert (alpha > 0.0).all()

    # `zeta` is the standard deviation of the prior over the means. A higher value corresponds to
    # the component means being more spread out.
    assert zeta > 0.0

    # `sigma` is the standard deviation of the data points when sampled from each component. A
    # higher value corresponds to the data points being more spread out from their component means.
    assert sigma > 0.0

    # `N` is the number of data points to sample.
    assert N > 0

    # Sample the component proportions `theta`.
    theta = random_state.dirichlet(alpha)
    assert theta.shape == (K,)

    # Sample the component means `mu`.
    mu = random_state.multivariate_normal(np.zeros(D), zeta**2 * np.eye(D), size=K)
    assert mu.shape == (K, D)

    # Sample the component assignments `z` and data points `X`.
    # Note that `m[k]` is the number of data points for component `k`.
    m = random_state.multinomial(N, theta)
    assert m.shape == (K,)

    z = np.repeat(np.arange(K), m)
    random_state.shuffle(z)
    assert z.shape == (N,)

    X = mu[z] + sigma * random_state.randn(N, D)

    return mu, theta, z, X


# Example.
alpha = [1.0, 1.0, 1.0]
zeta = 2.0
sigma = 1.0
print 'alpha:', alpha
print 'zeta:', zeta
print 'sigma:', sigma

D = 2
N = 1000


# Sample from the Gaussian Mixture Model.
random_state = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mu, theta, z, X = sample_gmm(alpha, zeta, sigma, D, N, random_state=random_state)


# Write `X` ...
print '-> X.tsv (X.shape: {})'.format(X.shape)
np.savetxt('X.tsv', X, delimiter='\t')
np.savetxt('mu.tsv', mu, delimiter='\t')
np.savetxt('z.tsv', z, delimiter='\t')

# ... and visualize.
fig, ax = plt.subplots()
ax.grid(True)
ax.set_aspect('equal')

prop_cycle = cycle(plt.rcParams['axes.prop_cycle'])

for k, (x, y) in enumerate(mu):
    p = next(prop_cycle)
    ax.plot(x, y, 'o', ms=8, mew=2.0, zorder=2.1, color=p['color'],
            marker=(3+k, 0, 0))

    x, y = X[z == k].T
    ax.plot(x, y, '.', color=p['color'])

plt.show()
