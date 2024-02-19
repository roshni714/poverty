from data_loaders.data_loader import load_dataset
from data_loaders.data_utils import standardize
from cond_dist import GLMConditionalDistribution

import statsmodels.api as sm
import numpy as np
import torch
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import tqdm


def get_semi_synthetic_malawi_data(n, d):
    np.random.seed(12345)
    torch.manual_seed(12345)
    _, y, r, _ = load_dataset("malawi")

    y = np.log(y)
    y, y_mean, y_std = standardize(y)

    kde = sm.nonparametric.KDEUnivariate(y)
    kde.fit(weights=r, fft=False, adjust=0.5)

    if d % 2 == 1:
        d_cont = int(d / 2) + 1
        d_discrete = int(d / 2)
    else:
        d_cont = int(d / 2)
        d_discrete = int(d / 2)
    X_cont = torch.Tensor(np.random.uniform(-1.0, 1.0, n * d_cont)).reshape(
        n, 1, d_cont
    )
    X_discrete = torch.Tensor(np.random.binomial(1, 0.5, n * d_discrete)).reshape(
        n, 1, d_discrete
    )
    X = torch.cat((X_cont, X_discrete), axis=2).reshape(n, d)
    beta = torch.Tensor(np.random.uniform(-1.0, 5.0, d).reshape(d, 1)) / d
    bin_ends = torch.linspace(min(y), max(y), 2000)

    gamma = torch.Tensor(np.random.uniform(-1.0, 3.0, d).reshape(d, 1)) / d

    def get_conditional_density(X_test):
        mu = torch.matmul(
            torch.tensor(X_test).reshape(X_test.shape[0], 1, d), beta
        ).reshape(X_test.shape[0], 1)
        sigma = torch.clamp(
            torch.matmul(
                torch.tensor(X_test).reshape(X_test.shape[0], 1, d), gamma
            ).reshape(X_test.shape[0], 1)
            ** 2,
            min=0.25,
            max=10.0,
        )

        front = torch.tensor(kde.evaluate(bin_ends.numpy()))

        unnormalized_densities = (front * torch.exp(-mu * bin_ends)).reshape(
            len(X_test), len(bin_ends)
        )
        norm_constant = torch.trapezoid(
            y=unnormalized_densities, x=bin_ends, axis=1
        ).unsqueeze(dim=1)
        pdf_matrix = unnormalized_densities / norm_constant

        unscaled_bin_ends = torch.exp(bin_ends * y_std + y_mean)

        pdf_matrix /= y_std * unscaled_bin_ends

        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal, axis=1)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal, axis=1)

        cdf_matrix = torch.cumulative_trapezoid(
            y=pdf_matrix, x=unscaled_bin_ends, dim=1
        )

        cond_dists = []

        best_idx = torch.argmax(pdf_matrix, axis=1)
        modes = unscaled_bin_ends[best_idx]

        for i in tqdm.tqdm(range(len(X_test))):
            idx_extrema = np.sort(
                np.hstack(
                    (
                        idx_maxima[1][idx_maxima[0] == i],
                        idx_minima[1][idx_minima[0] == i],
                    )
                )
            )

            cdf_function = interp1d(
                unscaled_bin_ends[1:],
                cdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )
            pdf_function = interp1d(
                unscaled_bin_ends,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=0.0,
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                unscaled_bin_ends[1:],
                bounds_error=False,
                fill_value=(unscaled_bin_ends[1], unscaled_bin_ends[-1]),
            )

            cond_dists.append(
                GLMConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=unscaled_bin_ends[idx_extrema],
                    outcome_range=(unscaled_bin_ends[0], unscaled_bin_ends[-1]),
                    mode=modes[i].item(),
                )
            )

        return np.array(cond_dists)

    dists = get_conditional_density(X.numpy())
    uniforms = torch.rand(len(X))
    sampled_y = np.array(
        [dists[i].ppf(uniform_rv) for i, uniform_rv in enumerate(uniforms)]
    )
    return X.numpy(), sampled_y, get_conditional_density
