import numpy as np
import torch
import tqdm
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.preprocessing import SplineTransformer

from opt_targeted_transfers.dataset_utils import standardize
from opt_targeted_transfers.cond_dist import NonparametricConditionalDistribution


def get_nll(dataset, cond_density_estimator):
    """
    Compute the negative log-likelihood of the conditional density using samples from dataset.

    :param dataset: The dataset for which to compute the negative log-likelihood.
    :type dataset: Dataset
    :param cond_density_estimator: The conditional density estimator.
    :type cond_density_estimator: Callable[[np.ndarray], np.ndarray]
    :return: The negative log-likelihood of the conditional density.
    """

    X, y, r = dataset.get_data()
    cond_dists = cond_density_estimator(X)

    nlls = []
    for i in range(len(y)):
        nlls.append(-np.log(cond_dists[i].pdf(y[i])) * r[i])
    return np.sum(nlls) / np.sum(r)


def get_cond_density_estimator(
    dataset, n_bins=100, n_knots=4, degree=4, truncation_upper_value=10, n_epochs=300
):
    """
    Compute the conditional density estimator.

    :param dataset: The dataset for which to compute the conditional density estimator.
    :type dataset: Dataset
    :param n_bins: The number of bins to use for the outcome space.
    :type n_bins: int
    :param n_knots: The number of knots to use for the B-spline basis functions.
    :type n_knots: int
    :param degree: The degree of the B-spline basis functions.
    :type degree: int
    :param truncation_upper_value: Truncate the outcome space at this value.
    :type truncation_upper_value: float
    :param n_epochs: The number of epochs to train the density estimator.
                     Defaults to 300.
    :type n_epochs: int
    :return: The conditional density estimator as a Python function that takes a numpy array
             with shape (N, D), where D is the same as the dimension of X in the dataset.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    if len(dataset.covs) == 0:
        helper = lindsey_method(
            dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
        )

    else:
        helper = lindsey_method_with_covariates(
            dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
        )
    return helper


def fit_carrier_function(y, r):
    """
    Computes the carrier density.

    :param y: A numpy array of Y values
    :type y: np.ndarray
    :param r: A numpy array of R values
    :type r: np.ndarray
    :return: A kernel density estimate of the Y-marginal.
    :rtype: KDEUnivariate
    """
    kde = KDEUnivariate(y)
    kde.fit(weights=r, fft=False, adjust=0.8)
    return kde


def lindsey_method(
    train_dataset,
    n_bins=100,
    n_knots=4,
    degree=3,
    truncation_upper_value=10,
    n_epochs=300,
    seed=123456,
):
    """
    Apply the Lindsey's method for marginal density estimation (Efron & Tibshirani 1996).

    :param train_dataset: The training dataset for which to apply the Lindsey method.
    :type train_dataset: Dataset
    :param n_bins: The number of bins to use for the outcome space.
    :type n_bins: int
    :param n_knots: The number of knots to use for the spline basis functions.
    :type n_knots: int
    :param degree: The degree of the spline basis functions.
    :type degree: int
    :param n_epochs: The number of epochs to train the density estimator.
                     Defaults to 300.
    :type n_epochs: int
    :param seed: The random seed to use for the method.
    :type seed: int
    :return: A callable that maps a numpy array to a numpy array of NonparametricConditionalDistribution objects.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get data. Truncate outcome because we only care about
    # accurate density estimation for units with Y < poverty line.
    # Standardize Y.
    _, y, r = train_dataset.get_data()
    n = len(y)
    y = np.clip(y, None, truncation_upper_value)
    y, y_mean, y_std = standardize(y)

    # Define range where learned density is well defined.
    lower = (0.0 - y_mean) / y_std
    upper = (truncation_upper_value - y_mean) / y_std

    # Bin the outcome space.
    bin_ends = np.linspace(lower, upper, n_bins)

    # Fit carrier density (Y marginal) and evaluate on bin boundaries.
    kde = fit_carrier_function(y, r)
    front = kde.evaluate(bin_ends)

    # Get B-spline basis functions.
    spline = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile")
    # This sets knots at evenly spaced quantiles of Y.
    spline.fit(y.reshape(-1, 1))

    # Get basis representation of bin boundaries.
    bin_basis_elements = spline.transform(bin_ends.reshape(-1, 1))
    k = bin_basis_elements.shape[1]
    bin_basis_elements = torch.tensor(
        bin_basis_elements.reshape(n_bins, k), dtype=torch.float64
    )  # n_bins x k

    # Get basis representation of sampled Y values.
    basis_matrix = torch.tensor(
        spline.transform(y.reshape(-1, 1)).reshape(n, k), dtype=torch.float64
    )  # n x k

    r = torch.tensor(r, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    bin_ends = torch.tensor(bin_ends, dtype=torch.float64)
    front = torch.tensor(front, dtype=torch.float64)
    theta = torch.nn.Parameter(
        torch.tensor(np.random.uniform(-1.0, 1.0, k).reshape(k, 1), dtype=torch.float64)
    )

    unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, idx):
        res = torch.matmul(basis_matrix[idx, :], theta).squeeze()  # n
        norm_res = torch.exp(
            torch.matmul(bin_basis_elements, theta)
        ).squeeze()  # n_bins  x1
        final_matrix = front * norm_res
        log_norm_constant = torch.log(torch.trapezoid(y=final_matrix, x=bin_ends))
        nll = -res + log_norm_constant
        return nll

    optimizer = torch.optim.Adam([theta], lr=1e-2)
    batch_size = int(len(y) / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))
    train_prop = 0.7
    idx_train_set, idx_val_set = list(range(int(train_prop * len(y)))), list(
        range(int(train_prop * len(y)), len(y))
    )
    thetas = []
    val_losses = []
    for epoch in pbar:
        if epoch % 25 == 0:
            val_loss = torch.sum(glm_nll(theta, idx_val_set) * r[idx_val_set])
            val_losses.append(val_loss.detach().item())
            thetas.append(theta.detach().clone())

        idx = np.random.choice(idx_train_set, size=batch_size)
        optimizer.zero_grad()
        loss = torch.sum(glm_nll(theta, idx) * r[idx])
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    best_model_idx = np.argmin(val_losses)
    final_theta = thetas[best_model_idx]
    print("Final Theta: {}".format(final_theta))

    def helper(X_test):
        norm_res = torch.exp(torch.matmul(bin_basis_elements, final_theta)).squeeze()
        final_matrix = front * norm_res
        norm_constant = torch.trapezoid(final_matrix, bin_ends)

        center = torch.exp(
            torch.matmul(
                bin_basis_elements,
                final_theta,
            )
        ).squeeze()
        pdf_matrix = (front * center) / norm_constant / y_std

        pdf_matrix = pdf_matrix.detach()

        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal)
        cdf_matrix = torch.cumulative_trapezoid(y=pdf_matrix, x=unscaled_bin_ends)

        cond_dists = []

        best_idx = torch.argmax(pdf_matrix)
        mode = unscaled_bin_ends[best_idx]
        idx_extrema = np.sort(
            np.hstack(
                (
                    idx_maxima,
                    idx_minima,
                )
            )
        ).flatten()

        cdf_function = interp1d(
            unscaled_bin_ends[1:],
            cdf_matrix,
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        pdf_function = interp1d(
            unscaled_bin_ends,
            pdf_matrix,
            bounds_error=False,
            fill_value=(0.0, 1e-6),
        )
        ppf_function = interp1d(
            cdf_matrix,
            unscaled_bin_ends[1:],
            bounds_error=False,
            fill_value=(unscaled_bin_ends[1], unscaled_bin_ends[-1]),
        )

        # assert np.isnan(cdf_matrix) is False
        # assert np.isnan(pdf_matrix) is False

        for i in range(len(X_test)):
            cond_dists.append(
                NonparametricConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=unscaled_bin_ends[idx_extrema].numpy(),
                    outcome_range=(unscaled_bin_ends[0], unscaled_bin_ends[-1]),
                    mode=mode.item(),
                )
            )

        return np.array(cond_dists)

    return helper


def lindsey_method_with_covariates(
    train_dataset,
    n_bins=100,
    n_knots=4,
    degree=3,
    truncation_upper_value=10,
    n_epochs=300,
    seed=123456,
):
    """
    Apply the Lindsey's method for marginal density estimation (Efron & Tibshirani 1996).

    :param train_dataset: The training dataset for which to apply the Lindsey method.
    :type train_dataset: Dataset
    :param n_bins: The number of bins to use for the outcome space.
    :type n_bins: int
    :param n_knots: The number of knots to use for the spline basis functions.
    :type n_knots: int
    :param degree: The degree of the spline basis functions.
    :type degree: int
    :param n_epochs: The number of epochs to train the density estimator.
                     Defaults to 300.
    :type n_epochs: int
    :return: A callable that maps a numpy array to a numpy array of NonparametricConditionalDistribution objects.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, r = train_dataset.get_data()

    # Get data. Truncate outcome because we only care about
    # accurate density estimation for units with Y < poverty line.
    # Standardize Y.
    _, y, r = train_dataset.get_data()
    n = len(y)
    y = np.clip(y, None, truncation_upper_value)
    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    # Define range where learned density is well defined.
    lower = (0.0 - y_mean) / y_std
    upper = (truncation_upper_value - y_mean) / y_std

    # Bin the outcome space.
    bin_ends = np.linspace(lower, upper, n_bins)

    n = X.shape[0]
    d = X.shape[1]

    # Fit carrier density (Y marginal) and evaluate on bin boundaries.
    kde = fit_carrier_function(y, r)
    front = kde.evaluate(bin_ends)

    # Get B-spline basis functions.
    spline = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile")
    # This sets knots at evenly spaced quantiles of Y.
    spline.fit(y.reshape(-1, 1))

    # Get basis representation of bin boundaries.
    bin_basis_elements = spline.transform(bin_ends.reshape(-1, 1))
    k = bin_basis_elements.shape[1]
    bin_basis_elements = torch.tensor(
        bin_basis_elements.reshape(n_bins, 1, k), dtype=torch.float64
    )  # n_bins x 1 x k

    # Get basis representation of sampled Y values.
    basis_matrix = torch.tensor(
        spline.transform(y.reshape(-1, 1)).reshape(n, 1, k), dtype=torch.float64
    )  # n x 1 x k

    X = torch.tensor(X, dtype=torch.float64).reshape(n, d, 1)
    r = torch.tensor(r, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    bin_ends = torch.tensor(bin_ends, dtype=torch.float64)
    front = torch.tensor(front, dtype=torch.float64)

    theta = torch.nn.Parameter(
        torch.tensor(
            np.random.uniform(-1.0, 1.0, k * d).reshape(k, d), dtype=torch.float64
        )
    )
    unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, idx):
        sub_n = len(idx)
        params = torch.matmul(theta, X[idx, :]).reshape(sub_n, k, 1)  # n x k x 1
        res = torch.matmul(basis_matrix[idx, :], params).squeeze()  # n x 1 x 1
        norm_res = torch.exp(
            torch.matmul(
                params.reshape(params.shape[0], params.shape[1]),
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ).T,
            )
        )  # n x J
        final_matrix = front * norm_res
        log_norm_constant = torch.log(
            torch.trapezoid(y=final_matrix, x=bin_ends, dim=1)
        )
        nll = -res + log_norm_constant
        return nll

    optimizer = torch.optim.Adam([theta], lr=1e-2)
    batch_size = int(len(X) / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))
    train_prop = 0.7
    idx_train_set, idx_val_set = list(range(int(train_prop * len(X)))), list(
        range(int(train_prop * len(X)), len(X))
    )
    thetas = []
    val_losses = []
    for epoch in pbar:
        if epoch % 25 == 0:
            val_loss = torch.sum(glm_nll(theta, idx_val_set) * r[idx_val_set])
            val_losses.append(val_loss.detach().item())
            thetas.append(theta.detach().clone())

        idx = np.random.choice(idx_train_set, size=batch_size)
        optimizer.zero_grad()
        loss = torch.sum(glm_nll(theta, idx) * r[idx])
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    best_model_idx = np.argmin(val_losses)
    final_theta = thetas[best_model_idx]
    print("Final Theta: {}".format(final_theta))

    def helper(X_test):
        X_test = (X_test - X_mean) / X_std
        nat_param = torch.matmul(
            final_theta,
            torch.tensor(X_test, dtype=torch.float64).reshape(
                X_test.shape[0], X_test.shape[1], 1
            ),
        )  # n x k x 1
        norm_res = torch.exp(
            torch.matmul(
                nat_param.reshape(nat_param.shape[0], nat_param.shape[1]),
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ).T,
            )
        )
        final_matrix = front * norm_res
        norm_constant = torch.trapezoid(final_matrix, bin_ends, dim=1)

        center = torch.exp(
            torch.matmul(
                nat_param.reshape(nat_param.shape[0], nat_param.shape[1]),
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ).T,
            )
        )
        pdf_matrix = ((front * center) / norm_constant.reshape(len(X_test), 1)) / y_std
        pdf_matrix = pdf_matrix.detach()

        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal, axis=1)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal, axis=1)

        cdf_matrix = torch.cumulative_trapezoid(
            y=pdf_matrix, x=unscaled_bin_ends, dim=1
        )
        cond_dists = []

        best_idx = torch.argmax(pdf_matrix, axis=1)
        modes = unscaled_bin_ends[best_idx]

        for i in range(len(X_test)):
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
                fill_value=(0.0, 1e-6),
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                unscaled_bin_ends[1:],
                bounds_error=False,
                fill_value=(unscaled_bin_ends[1], unscaled_bin_ends[-1]),
            )

            cond_dists.append(
                NonparametricConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=unscaled_bin_ends[idx_extrema].flatten(),
                    outcome_range=(unscaled_bin_ends[0], unscaled_bin_ends[-1]),
                    mode=modes[i].item(),
                )
            )
        return np.array(cond_dists)

    return helper
