import torch
import numpy as np
from scipy.stats import lognorm

from cond_dist import ConditionalDistribution


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

    gamma = torch.Tensor(np.random.uniform(0.1, 5, d).reshape(d, 1))
    gamma0 = np.random.uniform(0.1, 5, 1)

    psi = torch.Tensor(np.random.uniform(0.0, 1.0, d).reshape(d, 1)) / d
    psi0 = np.random.uniform(0.1, 1.0, 1) / d

    scales = torch.matmul(X, gamma).reshape(n, 1) + gamma0
    shapes = torch.matmul(X, psi).reshape(n, 1) + psi0

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
    return X, y, true_cond_densities, gamma.numpy(), gamma0, psi, psi0
