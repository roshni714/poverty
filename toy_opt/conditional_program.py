import numpy as np
import xgboost as xg

from data_loaders.data_utils import standardize
from reporting import write_result
from evaluate import post_transfer_metrics, empirical_post_transfer_metrics


def solve_conditional_program(p_xs, cond_dists, budget, c_bar):
    cost = []
    assignments = {x_idx: [] for x_idx in range(len(p_xs))}
    for i, cond_dist in enumerate(cond_dists):
        if cond_dist.cdf(c_bar) > budget:
            cost.append((c_bar - cond_dist.ppf(budget)))
            assignments[i] = [(c_bar - cond_dist.ppf(budget), 1.0)]
        else:
            cost.append(0.0)
            assignments[i] = [(0.0, 1.0)]

    total_cost = np.sum(np.array(cost) * p_xs)
    return assignments, total_cost


def solve_conditional_program_quantile_regression(
    train_dataset, budget, c_bar, title="sim"
):
    X = train_dataset.X
    y = train_dataset.y
    r = train_dataset.r

    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    q_hat = xg.XGBRegressor(
        objective="reg:quantileerror",
        max_depth=3,
        n_estimators=10,
        quantile_alpha=budget,
    ).fit(X, y, sample_weight=r)

    def t(X_test):
        X_test = (X_test - X.mean()) / X.std()
        quantile = q_hat.predict(X_test) * y_std + y_mean
        transfer = np.maximum(c_bar - quantile, 0)
        return transfer

    return t


#
#    assignments = {x_idx: [] for x_idx in range(len(test_dataset))}
#    for i in range(len(X_test))
#        assignments[i].append((transfer[i], 1.))

#    if true_cond_densities is not None:
#        result = post_transfer_metrics(true_cond_densities, r, optimal_policy, c_bar)
#    elif test_dataset is not None:
#        result = empirical_post_transfer_metrics(test_dataset, optimal_policy, c_bar)

#    results_file = "results/{}_conditional_program.csv".format(title)

#    write_result(results_file, result)
#    return optimal_policy, total_cost
