import numpy as np
import xgboost as xg

from data_loaders.data_utils import standardize


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

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    if X.shape[1] == 0:
        q_hat = np.quantile(y, budget).item()
    else:
        q_hat = xg.XGBRegressor(
            objective="reg:quantileerror",
            max_depth=3,
            n_estimators=10,
            quantile_alpha=budget,
        ).fit(X, y)

    def t(X_test):
        if isinstance(q_hat, float):
            quantile = (q_hat * y_std + y_mean) * np.ones(X_test.shape[0])
        else:
            X_test = (X_test - X.mean()) / X.std()
            quantile = q_hat.predict(X_test) * y_std + y_mean
        transfer = np.maximum(c_bar - quantile, 0)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        for i in range(len(X_test)):
            assignments[i].append((transfer[i], 1.0))
        return assignments

    return t
