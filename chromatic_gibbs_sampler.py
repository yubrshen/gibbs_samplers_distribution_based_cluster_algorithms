##########################################
# File: chromatic_gibbs_sampler.py       #
# Copyright Primer Technologies Inc 2017 #
##########################################

"""Usage: python chromatic_gibbs_sampler.py [input_filename]"""

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
def sample_theta(count_by_component, alpha, random_state):
    gamma = alpha + count_by_component
    theta = random_state.dirichlet(gamma)
    return theta
def sample_mu(z, X, K, D, N, sigma, zeta, random_state):
    zn = [[n for n in range(N) if z[n]==k] for k in range(K)]
    X_by_component = np.array([X[zn[k]] for k in range(K)])
    count_by_component = np.array(map(len, X_by_component))
    sum_by_component = np.array([sum(c) if (0 < len(c)) else np.zeros(D) for c in X_by_component])
    denominator = count_by_component + sigma/zeta
    mean_for_mu = sum_by_component/denominator[:, None]
    var_for_mu =[np.eye(D)*sigma*d for d in denominator]
    mu = [random_state.multivariate_normal(mean_for_mu[k], var_for_mu[k], size=1)[0] for k in range(K)]

    return np.array(mu), count_by_component

def sample_z(z, X, K, D, N, alpha, sigma, zeta, mu, theta, random_state):
    diff_n_k_squared = np.array([y*y for y in [X-mu[k] for k in range(K)]])
    diff_n_k_sum = np.sum(diff_n_k_squared, axis=2)
    p_xn_k = np.exp(-diff_n_k_sum/(2.0*sigma))/np.power(2.0*np.pi*sigma, D/2.0)

    p_tild_xn_k = p_xn_k*theta[:, None]

    p_tild_xn_k_sum_over_k = np.sum(p_tild_xn_k, axis=0)
    p_tild_xn_k_sum_over_k[p_tild_xn_k_sum_over_k == 0.0] = 1.0
    # no normalization when the denomenator is 0.0
    # to avoid div by 0.0, though it should not happen
    p_z_n_k = p_tild_xn_k/p_tild_xn_k_sum_over_k
    z_next = []
    for n in range(N):
        try:
            z_next.append(random_state.choice(K, p=p_z_n_k[:,n]))
        except ValueError as ex:
            z_next.append(random_state.choice(K, p=None))
            print("The exception: {}; The offending probabilities: {}; fix it by a guess by random uniform".format(ex, p_z_n_k[:,n]))
        else: # successful case
            pass
            # print("Indeed some correct probabilities: {}".format(p_z_n_k[:,n]))
    # z_next = np.array([random_state.choice(K, p=p_z_n_k[:,n]) for n in range(N)])
    return np.array(z_next)

def chromatic_gibbs_sampler(X, alpha, zeta, sigma, random_state=None):
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
        mu, count_by_component = sample_mu(z, X, K, D, N, sigma, zeta, random_state)

        # Sample theta
        theta = sample_theta(count_by_component, alpha, random_state)

        # Sample each `z`.
        z = sample_z(z, X, K, D, N, alpha, sigma, zeta, mu, theta, random_state)

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
sampler = chromatic_gibbs_sampler(X, alpha, zeta, sigma, random_state=0)
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
