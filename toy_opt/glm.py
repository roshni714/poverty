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
from scipy.signal import argrelmin, argrelmax, argrelextrema
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


def get_glm_spline_fit_helper(train_dataset, num_basis_elements=5, log_transform=True):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    X, X_mean, X_std = standardize(X)

    if log_transform:
        y = np.log(y)
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

    bins = np.linspace(min(y), max(y), n_bins)
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

    if log_transform:
        unscaled_bins = np.exp(bins * y_std + y_mean)
    else:
        unscaled_bins = bins * y_std + y_mean

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

        # Compute modes
        #        y_bins = torch.exp(torch.Tensor(unscaled_bins))
        y_bins = torch.Tensor(unscaled_bins)
        pdf_matrix = (
            front
            * torch.exp(
                torch.matmul(nat_param.squeeze(), bin_basis_elements.squeeze().T)
                - log_norm_constant.reshape(len(X_test), 1)
            )
        ) / y_std  # / y_bins

        if log_transform:
            pdf_matrix /= y_bins

        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal, axis=1)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal, axis=1)

        cdf_matrix = torch.cumulative_trapezoid(pdf_matrix, y_bins, dim=1)

        cond_dists = []
        y_midpoint_bins = np.array(
            [(y_bins[i] + y_bins[i + 1]) / 2 for i in range(len(y_bins) - 1)]
        )

        best_idx = torch.argmax(pdf_matrix, axis=1)
        modes = y_bins[best_idx]

        for i in range(len(X_test)):
            idx_extrema = np.sort(
                np.hstack(
                    (
                        idx_maxima[1][idx_maxima[0] == i],
                        idx_minima[1][idx_minima[0] == i],
                    )
                )
            )

            #            if len(idx_extrema) == 0:
            #                import pdb
            #                pdb.set_trace()

            cdf_function = interp1d(
                y_midpoint_bins,
                cdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )
            pdf_function = interp1d(
                y_bins,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=(pdf_matrix[i][0], pdf_matrix[i][-1]),
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                y_midpoint_bins,
                bounds_error=False,
                fill_value=(y_midpoint_bins[0], y_midpoint_bins[-1]),
            )

            cond_dists.append(
                GLMSplineConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=y_bins[idx_extrema],
                    outcome_range=(y_bins[0], y_bins[-1]),
                    mode=modes[i].item(),
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
