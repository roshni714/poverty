from glm import fit_carrier_function
from data_loaders.data_utils import standardize
import numpy as np
import torch
import tqdm
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from cond_dist import GLMConditionalDistribution


def setup_basis(y):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = np.array([np.quantile(y, q) for q in qs])

    def get_basis_function(z):
        basis = np.maximum(z.reshape(len(z), 1) - quantiles, 0).reshape(
            len(z), 1, len(qs)
        )
        return basis

    return get_basis_function, len(qs)


def get_glm_boundary_fit_helper(train_dataset, log_transform=True):
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

    get_basis_matrix, k = setup_basis(y)

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

    n_epochs = 1000
    optimizer = torch.optim.Adam([theta], lr=1e-1)
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

    if log_transform:
        unscaled_bin_ends = torch.exp(bin_ends * y_std + y_mean)
    else:
        unscaled_bin_ends = bin_ends * y_std + y_mean

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
