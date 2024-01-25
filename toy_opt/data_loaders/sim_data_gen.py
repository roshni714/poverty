import torch
import numpy as np
from scipy.stats import lognorm

from cond_dist import LogNormalConditionalDistribution


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

    if d % 2 == 1:
        d_cont = int(d / 2) + 1
        d_discrete = int(d / 2)
    else:
        d_cont = int(d / 2)
        d_discrete = int(d / 2)
    X_cont = torch.Tensor(np.random.uniform(0.0, 1.0, n * d_cont)).reshape(n, 1, d_cont)
    X_discrete = torch.Tensor(np.random.binomial(1, 0.5, n * d_discrete)).reshape(
        n, 1, d_discrete
    )
    X = torch.cat((X_cont, X_discrete), axis=2)

    mu = torch.Tensor(np.random.uniform(-0.5, 1.0, d).reshape(d, 1))
    mu0 = np.random.uniform(-0.5, 1.0, 1)

    mag_sort = torch.argsort(torch.abs(mu), dim=0, descending=True)
    mu = mu[mag_sort].reshape(d, 1)

    psi = torch.Tensor(np.random.uniform(-0.25, 0.5, d).reshape(d, 1)) / d
    psi0 = np.random.uniform(-0.25, 0.5, 1) / d

    mag_sort = torch.argsort(torch.abs(psi), dim=0, descending=True)
    psi = psi[mag_sort].reshape(d, 1)

    print("mu: {}".format(mu.flatten()))
    print("psi: {}".format(psi.flatten()))

    scales = np.exp(torch.matmul(X, mu).reshape(n, 1) + mu0)
    shapes = np.exp(torch.matmul(X, psi).reshape(n, 1) + psi0)

    X = X.reshape(n, d).numpy()
    y = lognorm.rvs(loc=torch.zeros((n, 1)), scale=scales, s=shapes).reshape(
        n,
    )

    def get_cond_densities(X_test):
        n = len(X_test)
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
                    loc=0.0, scale=scales[i].item(), shape=shapes[i].item()
                )
            )
        return np.array(true_cond_densities)

    return X, y, get_cond_densities
