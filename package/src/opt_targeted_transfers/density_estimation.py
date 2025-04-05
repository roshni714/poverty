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
    train_dataset,
    validation_dataset,
    n_bins=100,
    n_knots=4,
    degree=4,
    truncation_upper_value=10,
    n_epochs=300,
    device="cpu",
):
    """
    Compute the conditional density estimator.

    :param train_dataset: The training dataset for which to compute the conditional density estimator.
    :type train_dataset: Dataset
    :param validation_dataset: The validation dataset for which to compute the conditional density estimator.
    :type validation_dataset: Dataset
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
    if len(train_dataset.covs) == 0:
        helper = lindsey_method(
            train_dataset,
            validation_dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
            device=device,
        )

    else:
        helper = lindsey_method_with_covariates(
            train_dataset,
            validation_dataset,
            n_bins=n_bins,
            n_knots=n_knots,
            degree=degree,
            truncation_upper_value=truncation_upper_value,
            n_epochs=n_epochs,
            device=device,
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
    validation_dataset,
    n_bins=100,
    n_knots=4,
    degree=3,
    truncation_upper_value=10,
    n_epochs=300,
    seed=123456,
    device="cpu",
):
    """
    Apply the Lindsey's method for marginal density estimation (Efron & Tibshirani 1996).

    :param train_dataset: The training dataset for which to apply the Lindsey method.
    :type train_dataset: Dataset
    :param validation_dataset: The validation dataset for which to apply the Lindsey method.
    :type validation_dataset: Dataset
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
    _, y_train, r_train = train_dataset.get_data()
    _, y_val, r_val = validation_dataset.get_data()
    n_train = len(y_train)
    n_val = len(y_val)
    y_train = np.clip(y_train, None, truncation_upper_value)
    y_val = np.clip(y_val, None, truncation_upper_value)
    y_train, y_mean, y_std = standardize(y_train)
    y_val = (y_val - y_mean) / y_std

    # Define range where learned density is well defined.
    lower = (0.0 - y_mean) / y_std
    upper = (truncation_upper_value - y_mean) / y_std

    # Bin the outcome space.
    bin_ends = np.linspace(lower, upper, n_bins)

    # Fit carrier density (Y marginal) and evaluate on bin boundaries.
    kde = fit_carrier_function(y_train, r_train)
    front = kde.evaluate(bin_ends)
    print(front)
    # Get B-spline basis functions.
    spline = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile")
    # This sets knots at evenly spaced quantiles of Y.
    spline.fit(y_train.reshape(-1, 1))

    # Get basis representation of bin boundaries.
    bin_basis_elements = spline.transform(bin_ends.reshape(-1, 1))
    k = bin_basis_elements.shape[1]
    bin_basis_elements = torch.tensor(
        bin_basis_elements.reshape(n_bins, k), dtype=torch.float64
    ).to(
        device
    )  # n_bins x k

    # Get basis representation of sampled Y values.
    basis_matrix_train = torch.tensor(
        spline.transform(y_train.reshape(-1, 1)).reshape(n_train, k),
        dtype=torch.float64,
    )  # n x k

    basis_matrix_val = torch.tensor(
        spline.transform(y_val.reshape(-1, 1)).reshape(n_val, k), dtype=torch.float64
    )  # n x k

    r_train = torch.tensor(r_train, dtype=torch.float64)
    r_val = torch.tensor(r_val, dtype=torch.float64)
    bin_ends = torch.tensor(bin_ends, dtype=torch.float64).to(device)
    front = torch.tensor(front, dtype=torch.float64).to(device)
    theta = torch.nn.Parameter(
        torch.tensor(
            np.random.uniform(-1.0, 1.0, k).reshape(k, 1),
            dtype=torch.float64,
            device=device,
        )
    )

    unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, basis_matrix):
        basis_matrix = basis_matrix.to(device)
        res = torch.matmul(basis_matrix, theta).squeeze()  # n
        norm_res = torch.exp(
            torch.matmul(bin_basis_elements, theta)
        ).squeeze()  # n_bins  x1
        final_matrix = front * norm_res
        log_norm_constant = torch.log(torch.trapezoid(y=final_matrix, x=bin_ends))
        nll = -res + log_norm_constant
        return nll

    optimizer = torch.optim.Adam([theta], lr=1e-2)
    batch_size = int(len(y_train) / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))

    thetas = []
    val_losses = []
    for epoch in pbar:
        if epoch % 25 == 0:
            with torch.no_grad():
                val_loss = torch.sum(glm_nll(theta, basis_matrix_val) * r_val.to(device))
                val_losses.append(val_loss.detach().item())
                thetas.append(theta.detach().clone().cpu())

        idx = np.random.choice(len(y_train), size=batch_size)
        optimizer.zero_grad()
        loss = torch.sum(
            glm_nll(theta, basis_matrix_train[idx, :]) * r_train[idx].to(device)
        )
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    best_model_idx = np.argmin(val_losses)
    final_theta = thetas[best_model_idx]
    print("Final Theta: {}".format(final_theta))

    bin_basis_elements = bin_basis_elements.cpu()
    front = front.cpu()
    bin_ends = bin_ends.cpu()
    final_theta = final_theta.cpu()
    unscaled_bin_ends = unscaled_bin_ends.cpu()

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
    validation_dataset,
    n_bins=100,
    n_knots=4,
    degree=3,
    truncation_upper_value=10,
    n_epochs=300,
    seed=123456,
    device="cpu",
):
    """
    Apply the Lindsey's method for marginal density estimation (Efron & Tibshirani 1996).

    :param train_dataset: The training dataset for which to apply the Lindsey method.
    :type train_dataset: Dataset
    :param validation_dataset: The validation dataset for which to apply the Lindsey method.
    :type validation_dataset: Dataset
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

    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()

    # Get data. Truncate outcome because we only care about
    # accurate density estimation for units with Y < poverty line.
    # Standardize Y.
    n = len(y_train)
    y_train = np.clip(y_train, None, truncation_upper_value)
    y_val = np.clip(y_val, None, truncation_upper_value)
    X_train, X_mean, X_std = standardize(X_train)
    y_train, y_mean, y_std = standardize(y_train)
    X_val = (X_val - X_mean) / X_std
    y_val = (y_val - y_mean) / y_std

    # Define range where learned density is well defined.
    lower = (0.0 - y_mean) / y_std
    upper = (truncation_upper_value - y_mean) / y_std

    # Bin the outcome space.
    bin_ends = np.linspace(lower, upper, n_bins)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    d = X_train.shape[1]

    # Fit carrier density (Y marginal) and evaluate on bin boundaries.
    kde = fit_carrier_function(y_train, r_train)
    front = kde.evaluate(bin_ends)

    # Get B-spline basis functions.
    spline = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile")
    # This sets knots at evenly spaced quantiles of Y.
    spline.fit(y_train.reshape(-1, 1))

    # Get basis representation of bin boundaries.
    bin_basis_elements = spline.transform(bin_ends.reshape(-1, 1))
    k = bin_basis_elements.shape[1]
    bin_basis_elements = torch.tensor(
        bin_basis_elements.reshape(n_bins, 1, k), dtype=torch.float64
    ).to(
        device
    )  # n_bins x 1 x k

    # Get basis representation of sampled Y values.
    basis_matrix_train = torch.tensor(
        spline.transform(y_train.reshape(-1, 1)).reshape(n_train, 1, k),
        dtype=torch.float64,
    )  # n x 1 x k

    basis_matrix_val = torch.tensor(
        spline.transform(y_val.reshape(-1, 1)).reshape(n_val, 1, k), dtype=torch.float64
    )  # n x 1 x k

    X_train = torch.tensor(X_train, dtype=torch.float64).reshape(n_train, d, 1)
    r_train = torch.tensor(r_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)

    X_val = torch.tensor(X_val, dtype=torch.float64).reshape(n_val, d, 1)
    r_val = torch.tensor(r_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.float64)

    bin_ends = torch.tensor(bin_ends, dtype=torch.float64).to(device)
    front = torch.tensor(front, dtype=torch.float64).to(device)

    theta = torch.nn.Parameter(
        torch.tensor(
            np.random.uniform(-1.0, 1.0, k * d).reshape(k, d),
            dtype=torch.float64,
            device=device,
        )
    )

    unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, X, basis_matrix):
        sub_n = len(X)
        params = torch.matmul(theta, X.to(device)).reshape(sub_n, k, 1)  # n x k x 1
        res = torch.matmul(basis_matrix.to(device), params).squeeze()  # n x 1 x 1
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
    batch_size = int(n_train / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))

    thetas = []
    val_losses = []
    for epoch in pbar:
        if epoch % 25 == 0:
            val_loss = torch.sum(
                glm_nll(theta, X_val, basis_matrix_val) * r_val.to(device)
            )
            val_losses.append(val_loss.detach().item())
            thetas.append(theta.detach().clone().cpu())

        idx = np.random.choice(n_train, size=batch_size, replace=True)
        optimizer.zero_grad()
        loss = torch.sum(
            glm_nll(theta, X_train[idx, :], basis_matrix_train[idx, :])
            * r_train[idx].to(device)
        )
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    best_model_idx = np.argmin(val_losses)
    final_theta = thetas[best_model_idx]
    print("Final Theta: {}".format(final_theta))

    bin_basis_elements = bin_basis_elements.cpu()
    front = front.cpu()
    bin_ends = bin_ends.cpu()
    final_theta = final_theta.cpu()
    unscaled_bin_ends = unscaled_bin_ends.cpu()

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
