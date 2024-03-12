import numpy as np
import torch
import tqdm
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import statsmodels.gam.smooth_basis as sb
from statsmodels.nonparametric.kde import KDEUnivariate


from opt_targeted_transfers.dataset_utils import standardize
from opt_targeted_transfers.cond_dist import GLMConditionalDistribution

def get_cond_density_estimator(dataset, log_transform=True, knot_quantiles=None, n_epochs=300):
    if dataset.X.shape[1] == 0:
        helper = lindsey_method(dataset, log_transform, knot_quantiles, n_epochs)

    else:
        helper = lindsey_method_with_covariates(dataset, log_transform, knot_quantiles, n_epochs)
    return helper

def fit_carrier_function(y, r):
    kde = KDEUnivariate(y)
    kde.fit(weights=r, fft=False, adjust=0.8)
    return kde

def setup_bspline_basis(y, degree=3, knot_quantiles=None):
    # More knots at small quantiles

    if knot_quantiles is None:
        qs = [0.1, 0.20, 0.4, 0.6]
    else:
        qs = knot_quantiles

    internal_knots = [np.quantile(y, q) for q in qs]

    df = len(internal_knots) + degree + 1
    spline_matrix = sb.BSplines(
            y,
            df=df,
            degree=degree,
            include_intercept=True,
            knot_kwds=[{"knots": internal_knots}],
        )

    knots = spline_matrix.smoothers[0].knots
    num_basis_elem = spline_matrix.basis.shape[1]

    def get_basis(z):
        spline_matrix = sb.BSplines(
            z,
            df=df,
            degree=degree,
            include_intercept=True,
            knot_kwds=[{"all_knots": knots}],
        )

        basis = spline_matrix.basis.reshape(len(z), 1, num_basis_elem)
        return basis

    return get_basis, num_basis_elem

def lindsey_method(train_dataset, log_transform=True, knot_quantiles=None, n_epochs=300):
    y = train_dataset.y
    r = train_dataset.r

    if log_transform:
        y = np.log(y)
    y, y_mean, y_std = standardize(y)

    n = y.shape[0]
    torch.manual_seed(123456)
    np.random.seed(123456)

    n_bins = 2000
    bin_ends = np.linspace(min(y), max(y), n_bins)
    kde = fit_carrier_function(y, r)
    front = kde.evaluate(bin_ends)

    get_basis_matrix, k = setup_bspline_basis(y, degree=3, knot_quantiles=knot_quantiles)

    bin_basis_elements = get_basis_matrix(bin_ends)
    basis_matrix = get_basis_matrix(y)  # n x 1 x k

    print("Made basis")

    r = torch.tensor(r, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    basis_matrix = torch.tensor(basis_matrix, dtype=torch.float64)  # n x k
    bin_basis_elements = torch.tensor(
        bin_basis_elements, dtype=torch.float64
    )  # n_bins x k
    bin_ends = torch.tensor(bin_ends, dtype=torch.float64)
    front = torch.tensor(front, dtype=torch.float64)
    theta = torch.nn.Parameter(
        torch.tensor(np.random.uniform(-1.0, 1.0, k).reshape(k, 1), dtype=torch.float64)
    )
    if log_transform:
        unscaled_bin_ends = torch.exp(bin_ends * y_std + y_mean)
    else:
        unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, idx):
        sub_n = len(idx)
        res = torch.matmul(basis_matrix[idx, :], theta).squeeze()  # n
        norm_res = torch.exp(
            torch.matmul(
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ),
                theta,
            )
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
        norm_res = torch.exp(
            torch.matmul(
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ),
                final_theta,
            )
        ).squeeze()
        final_matrix = front * norm_res
        norm_constant = torch.trapezoid(final_matrix, bin_ends)

        center = torch.exp(
            torch.matmul(
                bin_basis_elements.reshape(
                    bin_basis_elements.shape[0], bin_basis_elements.shape[2]
                ),
                final_theta,
            )
        ).squeeze()
        pdf_matrix = (front * center) / norm_constant / y_std

        if log_transform:
            pdf_matrix = pdf_matrix / unscaled_bin_ends

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
        )
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
            fill_value=0.0,
        )
        ppf_function = interp1d(
            cdf_matrix,
            unscaled_bin_ends[1:],
            bounds_error=False,
            fill_value=(unscaled_bin_ends[1], unscaled_bin_ends[-1]),
        )

        for i in range(len(X_test)):
            cond_dists.append(
                GLMConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=unscaled_bin_ends[idx_extrema],
                    outcome_range=(unscaled_bin_ends[0], unscaled_bin_ends[-1]),
                    mode=mode.item(),
                )
            )

        return cond_dists

    return helper


def lindsey_method_with_covariates(train_dataset, log_transform=True, knot_quantiles=None, n_epochs=300):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    X, X_mean, X_std = standardize(X)

    if log_transform:
        y = np.log(y)
    y, y_mean, y_std = standardize(y)

    n = X.shape[0]
    d = X.shape[1]
    torch.manual_seed(123456)
    np.random.seed(123456)

    n_bins = 2000
    bin_ends = np.linspace(min(y), max(y), n_bins)
    kde = fit_carrier_function(y, r)
    front = kde.evaluate(bin_ends)

    get_basis_matrix, k = setup_bspline_basis(y, degree=3, knot_quantiles=knot_quantiles)
    bin_basis_elements = get_basis_matrix(bin_ends)
    basis_matrix = get_basis_matrix(y)  # n x 1 x k

    X = torch.tensor(X, dtype=torch.float64).reshape(n, d, 1)
    r = torch.tensor(r, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    basis_matrix = torch.tensor(basis_matrix, dtype=torch.float64)
    bin_basis_elements = torch.tensor(bin_basis_elements, dtype=torch.float64)
    bin_ends = torch.tensor(bin_ends, dtype=torch.float64)
    front = torch.tensor(front, dtype=torch.float64)
    theta = torch.nn.Parameter(
        torch.tensor(
            np.random.uniform(-1.0, 1.0, k * d).reshape(k, d), dtype=torch.float64
        )
    )
    if log_transform:
        unscaled_bin_ends = torch.exp(bin_ends * y_std + y_mean)
    else:
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
        if log_transform:
            pdf_matrix /= unscaled_bin_ends

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

    return helper
