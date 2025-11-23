import numpy as np
from opt_targeted_transfers.dataset_utils import standardize
from sklearn.linear_model import Lasso, LinearRegression
import torch
import tqdm
import copy


def get_pmt_lasso_regressor(train_dataset, validation_dataset, alpha=0.1):
    """
    Fit a Lasso regression model to the training data and return a function that predicts consumption.
    """
    from sklearn.linear_model import Lasso

    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    r = np.concatenate([r_train, r_val], axis=0)

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    if alpha == 0.0:
        # If alpha is 0, use Linear Regression instead of Lasso
        model = LinearRegression(fit_intercept=True)
    else:
        model = Lasso(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=r)

    def estimator(X_test):
        X_test = (X_test - X_mean) / X_std
        predicted_consumption = (
            model.predict(X_test).reshape(X_test.shape[0], 1) * y_std + y_mean
        )

        return predicted_consumption.flatten()

    return estimator


def get_pmt_linear_regressor(train_dataset, validation_dataset):
    # revise this to use cross validation in future?
    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    r = np.concatenate([r_train, r_val], axis=0)

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y, sample_weight=r)

    def estimator(X_test):
        X_test = (X_test - X_mean) / X_std
        predicted_consumption = (
            model.predict(X_test).reshape(X_test.shape[0], 1) * y_std + y_mean
        )

        return predicted_consumption.flatten()

    return estimator


def get_mse_loss(predictor, validation_dataset):
    X, y, r = validation_dataset.get_data()
    res = predictor(X)
    mse_loss = (res - y) ** 2
    weighted_mse_loss = np.sum(mse_loss * r) / np.sum(r)
    return weighted_mse_loss


def get_pmt_nn_regressor(
    train_dataset,
    validation_dataset,
    n_layers,
    n_hidden_units,
    lr,
    n_epochs=300,
    seed=123843,
    device="cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, r_train = train_dataset.get_data()
    X_val, y_val, r_val = validation_dataset.get_data()
    X_train, X_mean, X_std = standardize(X_train)
    y_train, y_mean, y_std = standardize(y_train)
    X_val = (X_val - X_mean) / X_std
    y_val = (y_val - y_mean) / y_std

    if X_train.shape[1] == 0:
        pass
    else:
        d = X_train.shape[1]

        model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            model_list.append(torch.nn.ReLU())
        model_list.append(torch.nn.Linear(n_hidden_units, 1))
        predictor = torch.nn.Sequential(*model_list).to(device)

        def mse_loss(predictor, X, y):
            y_pred = predictor(torch.Tensor(X).to(device)).squeeze()
            return (y_pred - torch.Tensor(y).to(device)) ** 2

        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        batch_size = int(len(X_train) / 5)
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 10 == 0:
                predictor.eval()

                val_loss = torch.sum(
                    mse_loss(predictor, X_val, y_val)
                    * torch.tensor(r_val).to(device)
                    / torch.tensor(r_val).sum().to(device)
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(predictor.cpu()))

            predictor.train()
            predictor = predictor.to(device)
            idx = np.random.choice(len(X_train), batch_size, replace=True)
            optimizer.zero_grad()
            loss = torch.sum(
                mse_loss(predictor, X_train[idx, :], y_train[idx])
                * torch.tensor(r_train[idx]).to(device)
            ) / torch.tensor(r_train[idx]).sum().to(device)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"val loss": val_losses[-1]})
        best_model_idx = np.argmin(val_losses)
        final_model = models[best_model_idx].cpu()

    def consumption_predictor(X_test):
        if X_test.shape[1] == 0:
            pass
        else:
            X_test = (X_test - X_mean) / X_std
            pred_consumption = (
                (
                    final_model(torch.Tensor(X_test)).reshape(X_test.shape[0], 1)
                    * y_std
                    + y_mean
                )
                .flatten()
                .detach()
                .numpy()
            )
            return pred_consumption

    return consumption_predictor
