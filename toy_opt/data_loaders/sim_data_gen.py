import torch
import numpy as np
from scipy.stats import lognorm

from cond_dist import LogNormalConditionalDistribution


def generate_toy_1d_data(n):
    X = np.linspace(0.0, 1.0, n).reshape(n, 1)

    loc = 3 * X + 4
    scale = 0.05 * X + 0.1
    shape = np.ones(X.shape)

    y = lognorm.rvs(loc=loc, scale=scale, s=shape).reshape(n)

    def get_cond_densities(X_test):
        n = len(X_test)
        locs = 3 * X_test + 4
        scales = 2 * X_test + 1
        shapes = np.ones(X_test.shape)

        true_cond_densities = []
        for i in range(n):
            true_cond_densities.append(
                LogNormalConditionalDistribution(
                    loc=locs[i].item(), scale=scales[i].item(), shape=shapes[i].item()
                )
            )
        return np.array(true_cond_densities)

    return X, y, get_cond_densities


def generate_homoscedastic_data(n, d):
    np.random.seed(12345)
    X = torch.Tensor(np.random.uniform(0.0, 1.0, n * d)).reshape(n, 1, d)

    gamma = torch.Tensor(np.random.uniform(2, 5, d).reshape(d, 1))
    gamma0 = np.random.uniform(2, 5, 1)

    scales = torch.matmul(X, gamma).reshape(n, 1) + gamma0
    shapes = torch.ones((n, 1))

    X = X.reshape(n, d).numpy()
    y = lognorm.rvs(loc=torch.zeros((n, 1)), scale=scales, s=shapes).reshape(
        n,
    )

    true_cond_densities = []
    for i in range(n):
        true_cond_densities.append(
            ConditionalDistribution(
                loc=0.0, scale=scales[i].item(), shape=shapes[i].item()
            )
        )
    return X, y, true_cond_densities, gamma.numpy(), gamma0


def generate_heteroscedastic_data(n, d):
    np.random.seed(12345)

    X = torch.Tensor(np.random.uniform(0.0, 1.0, n * d)).reshape(n, 1, d)

    #    beta = torch.Tensor(np.random.uniform(2, 6, d).reshape(d, 1)) / (d + 1e-5)
    #    beta0 = torch.Tensor(np.random.uniform(0.2, 1, 1))

    beta = torch.zeros((d, 1))
    beta0 = np.zeros(1)

    mu = torch.Tensor(np.random.uniform(-0.5, 1.0, d).reshape(d, 1))
    mu0 = np.random.uniform(-0.5, 1.0, 1)

    psi = torch.Tensor(np.random.uniform(-0.25, 0.5, d).reshape(d, 1)) / (d + 1e-5)
    psi0 = np.random.uniform(-0.05, 0.05, 1)

    print("beta: {}".format(beta.flatten()))
    print("mu: {}".format(mu.flatten()))
    print("psi: {}".format(psi.flatten()))

    locs = torch.matmul(X, beta).reshape(n, 1) + beta0
    scales = np.exp(torch.matmul(X, mu).reshape(n, 1) + mu0)
    shapes = np.exp(torch.matmul(X, psi).reshape(n, 1) + psi0)

    X = X.reshape(n, d).numpy()
    y = lognorm.rvs(loc=locs, scale=scales, s=shapes).reshape(
        n,
    )

    def get_cond_densities(X_test):
        n = len(X_test)
        locs = (
            torch.matmul(torch.Tensor(X_test).reshape(n, 1, d), beta).reshape(n, 1)
            + beta0
        )

        scales = np.exp(
            torch.matmul(torch.Tensor(X_test).reshape(n, 1, d), mu).reshape(n, 1) + mu0
        )
        shapes = np.exp(
            torch.matmul(torch.Tensor(X_test).reshape(n, 1, d), psi).reshape(n, 1)
            + psi0
        )
        true_cond_densities = []
        for i in range(n):
            true_cond_densities.append(
                LogNormalConditionalDistribution(
                    loc=locs[i].item(), scale=scales[i].item(), shape=shapes[i].item()
                )
            )
        return np.array(true_cond_densities)

    return X, y, get_cond_densities
