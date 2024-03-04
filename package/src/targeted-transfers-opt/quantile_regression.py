import torch
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import tqdm
import copy

from dataset_utils import standardize


def get_quantile_regressor(dataset, budget):
    X = dataset.X
    y = dataset.y
    r = dataset.r

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    np.random.seed(123456)
    torch.manual_seed(123456)

    if X.shape[1] == 0:
        wq = DescrStatsW(data=y, weights=r)
        final_q_hat = wq.quantile(budget).item()
    else:
        d = X.shape[1]
        q_hat = torch.nn.Sequential(
            torch.nn.Linear(d, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
        )

        def quantile_loss(q_hat, idx):
            sub_n = len(idx)
            y_pred = q_hat(torch.Tensor(X[idx, :])).squeeze()
            return budget * torch.nn.functional.relu(torch.Tensor(y[idx]) - y_pred) + (
                1 - budget
            ) * torch.nn.functional.relu(y_pred - torch.Tensor(y[idx]))

        n_epochs = 300
        optimizer = torch.optim.Adam(q_hat.parameters(), lr=1e-2)
        train_prop = 0.7
        idx_train_set, idx_val_set = list(range(int(train_prop * len(X)))), list(
            range(int(train_prop * len(X)), len(X))
        )

        batch_size = int(len(idx_train_set) / 3)
        print("Fitting conditional program - QR method via nonparametric regression...")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        val_losses = []
        models = []

        for epoch in pbar:
            if epoch % 25 == 0:
                val_loss = torch.sum(
                    quantile_loss(q_hat, idx_val_set) * torch.Tensor(r[idx_val_set])
                )
                val_losses.append(val_loss.detach().item())
                models.append(copy.deepcopy(q_hat))

            idx = np.random.choice(idx_train_set, size=batch_size)
            optimizer.zero_grad()
            loss = torch.sum(quantile_loss(q_hat, idx) * torch.Tensor(r[idx]))
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
        best_model_idx = np.argmin(val_losses)
        final_q_hat = models[best_model_idx]

        def quantile_regressor(X_test):
            if X_test.shape[1] == 0:
                quantile = (final_q_hat * y_std + y_mean) * np.ones(X_test.shape[0])
            else:
                X_test = (X_test - X_mean) / X_std
                quantile = (
                    (final_q_hat(torch.Tensor(X_test)).squeeze() * y_std + y_mean)
                    .detach()
                    .numpy()
                )
            return quantile

        return quantile_regressor
