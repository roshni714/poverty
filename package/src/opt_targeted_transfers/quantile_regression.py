import torch
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import tqdm
import copy

from opt_targeted_transfers.dataset_utils import standardize


def get_quantile_loss(validation_dataset, quantile_regressor, quantile):
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
    X, y, r = validation_dataset.get_data()
    y_pred = quantile_regressor(X)
    assert y_pred.shape == y.shape
    pinball_loss = quantile * np.maximum(y - y_pred, 0) + (1 - quantile) * np.maximum(
        y_pred - y, 0
    )
    assert pinball_loss.shape == r.shape
    weighted_pinball_loss = np.sum(pinball_loss * r) / np.sum(r)
    return weighted_pinball_loss


def get_quantile_regressor(
    train_dataset,
    validation_dataset,
    quantile,
    winsorize_outcome=97,
    n_layers=1,
    n_hidden_units=256,
    lr=5e-3,
    n_epochs=300,
    seed=123456,
    device="cpu",
):
    """
    Get a quantile regressor for a given dataset.

    :param train_dataset: The dataset used for training the quantile regressor.
    :type train_dataset: Dataset
    :param validation_dataset: The dataset used for evaluating the quantile regressor.
    :type validation_dataset: Dataset
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
    :param seed: The random seed.
    :type seed: int
    :param device: The device on which to train the neural network.
    :type device: str
    :return: The quantile regressor.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()
    upper_cap = np.percentile(y_train, winsorize_outcome)

    # Note: Assumes outcome is non-negative.
    y_train = np.clip(y_train, 0.0, upper_cap)
    y_val = np.clip(y_val, 0.0, upper_cap)
    X_train, X_mean, X_std = standardize(X_train)
    y_train, y_mean, y_std = standardize(y_train)
    X_val = (X_val - X_mean) / X_std
    y_val = (y_val - y_mean) / y_std

    if X_train.shape[1] == 0:
        wq = DescrStatsW(data=y, weights=r)
        final_q_hat = wq.quantile(quantile).item()
    else:
        d = X_train.shape[1]

        model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            model_list.append(torch.nn.ReLU())
        model_list.append(torch.nn.Linear(n_hidden_units, 1))
        q_hat = torch.nn.Sequential(*model_list).to(device)

        def quantile_loss(q_hat, X, y):
            y_pred = q_hat(torch.Tensor(X).to(device)).squeeze()
            return quantile * torch.nn.functional.relu(
                torch.Tensor(y).to(device) - y_pred
            ) + (1 - quantile) * torch.nn.functional.relu(
                y_pred - torch.Tensor(y).to(device)
            )

        optimizer = torch.optim.Adam(q_hat.parameters(), lr=lr)
        batch_size = int(min(16000, len(X_train)) / 5)
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 10 == 0:
                q_hat.eval()

                val_loss = torch.sum(
                    quantile_loss(q_hat, X_val, y_val)
                    * torch.tensor(r_val).to(device)
                    / torch.tensor(r_val).sum().to(device)
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(q_hat.cpu()))

            q_hat.train()
            q_hat = q_hat.to(device)
            idx = np.random.choice(len(X_train), batch_size, replace=True)
            optimizer.zero_grad()
            loss = torch.sum(
                quantile_loss(q_hat, X_train[idx, :], y_train[idx])
                * torch.tensor(r_train[idx]).to(device)
            ) / torch.tensor(r_train[idx]).sum().to(device)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"val loss": val_losses[-1]})
        best_model_idx = np.argmin(val_losses)
        final_q_hat = models[best_model_idx].cpu()

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
