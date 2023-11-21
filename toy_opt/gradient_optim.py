import numpy as np
import cvxpy as cp
from scipy.stats import norm
import scipy


K = 100
t0 = cp.Parameter(K, nonneg=True)
delta = cp.Variable(K)
x = np.linspace(0, 1, K)
eps = 0.2
c_bar = 0.3

eps_prime = cp.Parameter(1, nonneg=True)
f = cp.Parameter(K, nonneg=True)


def error_prob(c_bar, t):
    res = scipy.integrate.simps(
        norm.cdf(c_bar - t - np.linspace(0, 1, K)), np.linspace(0, 1, K)
    )
    return res


obj = cp.Minimize(cp.sum(t0 + delta) * (1 / K))
constraints = [t0 + delta >= 0, (-f @ delta) * (1 / K) <= eps_prime]
prob = cp.Problem(obj, constraints)

M = 1000
t0.value = np.ones(K) * 1.5
eps_prime.value = np.array([eps - error_prob(c_bar, t0.value)])
f.value = norm.pdf(-x)

costs = []
error_ps = []
etas = []


def backtracking_line_search(t_prev, delta, alpha=0.1, beta=0.9):
    eta = 1.0
    while error_prob(c_bar, t_prev + eta * delta) > eps:
        eta *= beta
    return eta


for i in range(M):
    prob.solve()
    cost = prob.value

    eta = backtracking_line_search(t0.value, delta.value)
    t = t0.value + eta * delta.value
    error_p = error_prob(c_bar, t)
    costs.append(cost)
    error_ps.append(error_p)
    etas.append(eta)
    #    print(delta.value)
    # update parameters
    t0.value = t
    f.value = norm.pdf(c_bar - t - x)
    eps_prime.value = np.array([np.maximum(eps - error_p, 0)])


# print(costs)
# print(error_ps)
# print(etas)
print(t)
