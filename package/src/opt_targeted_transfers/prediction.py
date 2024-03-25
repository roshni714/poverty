import torch
import numpy as np
import tqdm
import copy
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

from opt_targeted_transfers.cond_dist import NonparametricConditionalDistribution
from opt_targeted_transfers.dataset_utils import standardize


def get_prediction_function(dataset, n_classes, class_thresholds, n_epochs=300):
    """
    Get prediction function for a given dataset.

    :param dataset: The dataset used for training the regressor.
    :type dataset: Dataset
    :param n_epochs: The number of epochs for training the regressor. Defaults to 300.
    :type n_epochs: int
    :return: The quantile regressor.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    X = dataset.X
    y = dataset.y
    r = dataset.r

    X, X_mean, X_std = standardize(X)

    np.random.seed(123456)
    torch.manual_seed(123456)

    if X.shape[1] > 0:
        d = X.shape[1]
        h_hat = torch.nn.Sequential(
            torch.nn.Linear(d, 5), torch.nn.ReLU(), torch.nn.Linear(5, n_classes)
        )

        loss_f = torch.nn.CrossEntropyLoss(reduction="none")

        def cross_entropy_loss(h_hat, idx):
            sub_n = len(idx)
            pred = h_hat(torch.Tensor(X[idx, :]))
            return loss_f(pred, torch.Tensor(y[idx]).long())

        optimizer = torch.optim.Adam(h_hat.parameters(), lr=1e-2)
        train_prop = 0.7
        idx_train_set, idx_val_set = list(range(int(train_prop * len(X)))), list(
            range(int(train_prop * len(X)), len(X))
        )

        batch_size = int(len(idx_train_set) / 3)
        print("Fitting predictor via classification with cross entropy loss...")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 25 == 0:
                val_loss = torch.sum(
                    cross_entropy_loss(h_hat, idx_val_set)
                    * torch.Tensor(r[idx_val_set])
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(h_hat))

            idx = np.random.choice(idx_train_set, size=batch_size)
            optimizer.zero_grad()
            loss = torch.sum(cross_entropy_loss(h_hat, idx) * torch.Tensor(r[idx]))
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
        best_model_idx = np.argmin(val_losses)
        final_h_hat = models[best_model_idx]

    def prediction_function(X_test):
        if X_test.shape[1] == 0:
            pdf_matrix = torch.zeros(len(X_test), n_classes)
            pdf_matrix[:, 0] = 1.0
        else:
            X_test = (X_test - X_mean) / X_std
            pdf_matrix = (
                torch.nn.functional.softmax(
                    final_h_hat(torch.Tensor(X_test)), dim=1
                ).reshape(X_test.shape[0], n_classes)
            ).detach()
        cdf_matrix = torch.cumsum(pdf_matrix, dim=1)
        zeros = torch.zeros((len(X_test), 1))
        cdf_matrix = torch.cat((zeros, cdf_matrix), dim=1)[:, :-1]
        idx_maxima = argrelextrema(pdf_matrix.numpy(), np.less_equal, axis=1)
        idx_minima = argrelextrema(pdf_matrix.numpy(), np.greater_equal, axis=1)

        best_idx = torch.argmax(pdf_matrix, axis=1)
        modes = class_thresholds[best_idx]
        cond_dists = []

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
                class_thresholds,
                cdf_matrix[i].flatten(),
                bounds_error=False,
                kind="previous",
                fill_value=(0.0, 1.0),
            )
            pdf_function = interp1d(
                class_thresholds,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                kind="previous",
                fill_value=0.0,
            )
            ppf_function = interp1d(
                cdf_matrix[i].flatten(),
                class_thresholds,
                bounds_error=False,
                kind="previous",
                fill_value=(class_thresholds[0], class_thresholds[-1]),
            )

            cond_dists.append(
                NonparametricConditionalDistribution(
                    pdf_function,
                    cdf_function,
                    ppf_function,
                    extrema=class_thresholds[idx_extrema].flatten(),
                    outcome_range=(class_thresholds[0], class_thresholds[-1]),
                    mode=modes[i].item(),
                )
            )

        return cond_dists

    return prediction_function
