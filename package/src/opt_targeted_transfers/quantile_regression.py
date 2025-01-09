import torch
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import tqdm
import copy

from opt_targeted_transfers.dataset_utils import standardize


def get_quantile_loss(val_dataset, quantile_regressor, quantile):
    """
    Get the pinball loss for a given quantile regressor.

    :param val_dataset: The dataset used for evaluating the quantile regressor.
    :type val_dataset: Dataset
    :param quantile_regressor: The quantile regressor.
    :type quantile_regressor: Callable[[np.ndarray], np.ndarray]
    :param quantile: The quantile for which the pinball loss is computed.
    :type quantile: float
    :return: The pinball loss.
    :rtype: float
    """
    X, y, r = val_dataset.get_data()
    y_pred = quantile_regressor(X)
    assert y_pred.shape == y.shape
    pinball_loss = quantile * np.maximum(y - y_pred, 0) + (1 - quantile) * np.maximum(
        y_pred - y, 0
    )
    assert pinball_loss.shape == r.shape
    weighted_pinball_loss = np.sum(pinball_loss * r) / np.sum(r)
    return weighted_pinball_loss


def get_quantile_regressor(
    dataset, quantile, n_layers=1, n_hidden_units=256, lr=5e-3, n_epochs=300, seed=123456
):
    """
    Get a quantile regressor for a given dataset.

    :param dataset: The dataset used for training the quantile regressor.
    :type dataset: Dataset
    :param quantile: The quantile for which the regressor is trained.
    :type quantile: float
    :param n_layers: The number of hidden layers in the neural network.
    :type n_layers: int
    :param n_hidden_units: The number of hidden units in each hidden layer.
    :type n_hidden_units: int
    :param lr: The learning rate for training the neural network.
    :type lr: float
    :param n_epochs: The number of epochs for training the neural network.
    :type n_epochs: int
    :return: The quantile regressor.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, r = dataset.get_data()
    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    if X.shape[1] == 0:
        wq = DescrStatsW(data=y, weights=r)
        final_q_hat = wq.quantile(quantile).item()
    else:
        d = X.shape[1]

        model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            model_list.append(torch.nn.ReLU())
        model_list.append(torch.nn.Linear(n_hidden_units, 1))
        q_hat = torch.nn.Sequential(*model_list)

        def quantile_loss(q_hat, idx):
            y_pred = q_hat(torch.Tensor(X[idx, :])).squeeze()
            return quantile * torch.nn.functional.relu(
                torch.Tensor(y[idx]) - y_pred
            ) + (1 - quantile) * torch.nn.functional.relu(y_pred - torch.Tensor(y[idx]))

        optimizer = torch.optim.Adam(q_hat.parameters(), lr=lr)
        train_prop = 0.7
        idx_train_set, idx_val_set = list(range(int(train_prop * len(X)))), list(
            range(int(train_prop * len(X)), len(X))
        )

        batch_size = int(len(idx_train_set) / 5)
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 10 == 0:
                q_hat.eval()
                val_loss = torch.sum(
                    quantile_loss(q_hat, idx_val_set) * torch.Tensor(r[idx_val_set])
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(q_hat))

            q_hat.train()
            idx = np.random.choice(idx_train_set, size=batch_size)
            optimizer.zero_grad()
            loss = torch.sum(quantile_loss(q_hat, idx) * torch.Tensor(r[idx]))
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"val loss": val_losses[-1]})
        best_model_idx = np.argmin(val_losses)
        final_q_hat = models[best_model_idx]

    def quantile_regressor(X_test):
        if X_test.shape[1] == 0:
            quantile = (final_q_hat * y_std + y_mean) * np.ones((X_test.shape[0], 1))
        else:
            X_test = (X_test - X_mean) / X_std
            quantile = (
                (
                    final_q_hat(torch.Tensor(X_test)).reshape(X_test.shape[0], 1)
                    * y_std
                    + y_mean
                )
                .flatten()
                .detach()
                .numpy()
            )
            return np.maximum(quantile, 0.0)

    return quantile_regressor
