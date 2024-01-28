import numpy as np
from data_loaders.data_utils import standardize
from data_loaders.sim_data_gen import generate_heteroscedastic_data
import torch
from statsmodels.gam.smooth_basis import BSplines
import statsmodels.api as sm
from data_loaders.data_utils import Dataset
from cond_dist import GLMSplineConditionalDistribution
from scipy.stats import gaussian_kde
import tqdm
from scipy.interpolate import interp1d

import dill as pickle


def fit_carrier_function(y):
    kde = sm.nonparametric.KDEUnivariate(y)
    kde.fit()
    carrier_function = interp1d(
        kde.support,
        kde.density,
        bounds_error=False,
        fill_value=(kde.density[0], kde.density[-1]),
    )
    #    kde.set_bandwidth(bw_method=.4)
    return carrier_function


def get_basis_matrix(y, num_basis_elements):
    spline_matrix = BSplines(y, num_basis_elements + 1, num_basis_elements)
    return spline_matrix.basis.reshape(len(y), 1, num_basis_elements)


def get_glm_spline_fit_helper(train_dataset, outcome_range, num_basis_elements=6):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    carrier_function = fit_carrier_function(y)
    n = X.shape[0]
    d = X.shape[1]
    torch.manual_seed(123456)
    np.random.seed(123456)

    X = torch.Tensor(X).reshape(n, d, 1)
    r = torch.Tensor(r)
    k = num_basis_elements
    theta = torch.nn.Parameter(
        torch.Tensor(np.random.uniform(-1.0, 1.0, num_basis_elements * d).reshape(k, d))
    )
    basis_matrix = torch.Tensor(get_basis_matrix(y, k))  # n x 1 x k
    n_bins = 5000
    scaled_outcome_range = (np.array(outcome_range) - y_mean) / y_std
    bins = np.linspace(scaled_outcome_range[0], scaled_outcome_range[1], n_bins)
    bin_basis_elements = torch.Tensor(get_basis_matrix(bins, k))  # 500 x 1 x K
    front = torch.Tensor(carrier_function(bins))
    bin_width = bins[1] - bins[0]

    def glm_nll(theta, idx):
        sub_n = len(idx)
        params = torch.matmul(theta, X[idx, :]).reshape(sub_n, k, 1)  # n x k x 1
        res = torch.matmul(basis_matrix[idx, :], params).squeeze()  # n x 1 x 1
        # normalization constant
        norm_res = torch.exp(
            torch.matmul(params.squeeze(), bin_basis_elements.squeeze().T)
        )  # n x J
        final_matrix = front * norm_res
        log_norm_constant = torch.log(torch.sum(final_matrix, axis=1) * bin_width)

        nll = -res + log_norm_constant
        return nll

    n_epochs = 500
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    batch_size = int(len(X) / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))
    for epoch in pbar:
        idx = np.random.choice(len(X), size=batch_size)
        optimizer.zero_grad()

        loss = torch.sum(glm_nll(theta, idx) * r[idx])

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
        log_norm_constant = torch.log(torch.sum(final_matrix, axis=1) * bin_width)

        def spline_basis(y_test):
            y_test = (y_test - y_mean) / y_std
            return get_basis_matrix(y_test, k)

        # Compute modes
        unscaled_bins = torch.linspace(outcome_range[0], outcome_range[1], n_bins)
        scale_factor = (outcome_range[1] - outcome_range[0]) / (max(y) - min(y))
        pdf_matrix = (
            front
            * torch.exp(
                torch.matmul(nat_param.squeeze(), bin_basis_elements.squeeze().T)
                - log_norm_constant.reshape(len(X_test), 1)
            )
            / scale_factor
        )
        best_idx = np.argmax(pdf_matrix, axis=1)
        mode = unscaled_bins[best_idx]
        cdf_matrix = torch.cumulative_trapezoid(pdf_matrix, unscaled_bins, dim=1)

        cond_dists = []
        unscaled_midpoint_bins = np.array(
            [
                (unscaled_bins[i] + unscaled_bins[i + 1]) / 2
                for i in range(len(unscaled_bins) - 1)
            ]
        )
        for i in range(len(X_test)):
            cdf_function = interp1d(
                unscaled_midpoint_bins,
                cdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )
            pdf_function = interp1d(
                unscaled_bins,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=1e-5,
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                unscaled_midpoint_bins,
                bounds_error=False,
                fill_value=(min(y), max(y)),
            )

            cond_dists.append(
                GLMSplineConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    outcome_range,
                    mode[i].item(),
                )
            )

        return np.array(cond_dists)

    return helper


"""
X, y, _ = generate_heteroscedastic_data(n=1000, d=2)
train_dataset = Dataset(X, y, d=2)
estimator = get_glm_spline_fit_helper(train_dataset, 5)
cond_dists = estimator(X)
cond_dists[0].pdf(np.linspace(min(y), max(y)))
"""
