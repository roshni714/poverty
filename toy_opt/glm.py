import numpy as np
from data_loaders.data_utils import standardize
from data_loaders.sim_data_gen import generate_heteroscedastic_data
import torch
from statsmodels.gam.smooth_basis import BSplines
from data_loaders.data_utils import Dataset
from cond_dist import GLMSplineConditionalDistribution
from scipy.stats import gaussian_kde
import tqdm


def fit_carrier_function(y):
    kde = gaussian_kde(y)
    return kde


def get_basis_matrix(y, num_basis_elements):
    spline_matrix = BSplines(y, num_basis_elements + 1, num_basis_elements)
    return spline_matrix.basis.reshape(len(y), 1, num_basis_elements)


def fit_glm_spline_density(train_dataset, num_basis_elements=5):
    X = train_dataset.X
    y = train_dataset.y

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    carrier_function = fit_carrier_function(y)
    n = X.shape[0]
    d = X.shape[1]

    X = torch.Tensor(X).reshape(n, d, 1)
    k = num_basis_elements
    theta = torch.nn.Parameter(
        torch.Tensor(np.random.uniform(-1.0, 1.0, num_basis_elements * d).reshape(k, d))
    )
    basis_matrix = torch.Tensor(get_basis_matrix(y, k))  # n x 1 x k
    n_bins = 500
    y_range = np.array([min(y), max(y)])
    bins = np.linspace(min(y), max(y), n_bins)
    bin_basis_elements = torch.Tensor(
        get_basis_matrix(bins, num_basis_elements)
    )  # 500 x 1 x K
    front = torch.Tensor(carrier_function(bins))

    def glm_nll(theta, idx):
        sub_n = len(idx)
        params = torch.matmul(theta, X[idx, :]).reshape(sub_n, k, 1)  # n x k x 1
        res = torch.matmul(basis_matrix[idx, :], params)  # n x 1 x 1
        first_part = -torch.mean(res)

        # normalization constant
        norm_res = torch.exp(
            torch.matmul(params.squeeze(), bin_basis_elements.squeeze().T)
        )  # n x J
        final_matrix = front * norm_res
        norm_constant = torch.mean(final_matrix)

        nll = first_part + norm_constant
        return nll

    n_epochs = 200
    optimizer = torch.optim.Adam([theta], lr=1e-1)
    batch_size = int(len(X) / 5)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))
    for epoch in pbar:
        idx = np.random.choice(len(X), size=batch_size)
        optimizer.zero_grad()
        loss = glm_nll(theta, idx)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    final_theta = theta.detach().cpu()
    print("Final Theta: {}".format(final_theta))

    def helper(X_test):
        X_test = (X_test - X_mean) / X_std
        nat_param = torch.matmul(
            final_theta,
            torch.Tensor(X_test).reshape(X_test.shape[0], X_test.shape[1], 1),
        )  # n x k x 1
        norm_res = torch.exp(
            torch.matmul(nat_param.squeeze(), bin_basis_elements.squeeze().T)
        )
        final_matrix = front * norm_res
        norm_constant = torch.mean(final_matrix, axis=1)

        def spline_basis(y_test):
            y_test = (y_test - y_mean) / y_std
            return get_basis_matrix(y_test, k)

        def carrier(y_test):
            y_test = (y_test - y_mean) / y_std
            return carrier_function(y_test)

        cond_dists = []
        y_range_new = y_range * y_std + y_mean
        for i in range(len(X_test)):
            cond_dists.append(
                GLMSplineConditionalDistribution(
                    nat_param[i],
                    spline_basis,
                    carrier,
                    norm_constant[i].item(),
                    y_range_new,
                )
            )
        return np.array(cond_dists)

    return helper


X, y, cond_density_true = generate_heteroscedastic_data(10000, 2)
train_dataset = Dataset(X, y, d=2)
helper = fit_glm_spline_density(train_dataset)

cond_densities = helper(X[:100])
print(cond_densities[0].pdf(np.linspace(min(y), max(y), 100)))
