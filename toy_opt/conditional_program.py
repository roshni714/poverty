import numpy as np
import xgboost as xg

from data_loaders.data_utils import standardize
from statsmodels.stats.weightstats import DescrStatsW
import tqdm
import torch


def solve_conditional_program(compute_cond_density, budget, c_bar):
    def t(X_test):
        cond_densities = compute_cond_density(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        for i, cond_dist in enumerate(cond_densities):
            if cond_dist.cdf(c_bar) > budget:
                assignments[i] = [(c_bar - cond_dist.ppf(budget), 1.0)]
            else:
                assignments[i] = [(0.0, 1.0)]
        return assignments

    return t


def solve_conditional_program_quantile_regression(train_dataset, budget, c_bar):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    np.random.seed(123456)
    torch.manual_seed(123456)

    if X.shape[1] == 0:
        wq = DescrStatsW(data=y, weights=r)
        q_hat = wq.quantile(budget).item()
    else:
        d = X.shape[1]
        q_hat = torch.nn.Sequential(torch.nn.Linear(d, 5), torch.nn.Linear(5, 1))
        #        theta = torch.nn.Parameter(
        #        torch.Tensor(np.random.uniform(-1.0, 1.0, d).reshape(1, d))
        #        )

        def quantile_loss(q_hat, idx):
            sub_n = len(idx)
            y_pred = q_hat(torch.Tensor(X[idx, :])).squeeze()
            return budget * torch.nn.functional.relu(torch.Tensor(y[idx]) - y_pred) + (
                1 - budget
            ) * torch.nn.functional.relu(y_pred - torch.Tensor(y[idx]))

        n_epochs = 500
        optimizer = torch.optim.Adam(q_hat.parameters(), lr=1e-2)
        batch_size = int(len(X) / 3)
        print("Fitting conditional program - QR method via glm spline method...")
        pbar = tqdm.tqdm(list(range(n_epochs)))
        for epoch in pbar:
            idx = np.random.choice(len(X), size=batch_size)
            optimizer.zero_grad()
            loss = torch.sum(quantile_loss(q_hat, idx) * torch.Tensor(r[idx]))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    def t(X_test):
        if X_test.shape[1] == 0:
            quantile = (q_hat * y_std + y_mean) * np.ones(X_test.shape[0])
        else:
            X_test = (X_test - X_mean) / X_std
            quantile = (
                (q_hat(torch.Tensor(X_test)).squeeze() * y_std + y_mean)
                .detach()
                .numpy()
            )
        transfer = np.maximum(c_bar - quantile, 0)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        for i in range(len(X_test)):
            assignments[i].append((transfer[i], 1.0))
        return assignments

    return t
