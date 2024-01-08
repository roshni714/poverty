import numpy as np
from sklearn.linear_model import QuantileRegressor

from utils import standardize
from knapsack import policy_cost, prob_below_line
from reporting import write_result


def solve_conditional_program(p_xs, cond_dists, budget, c_bar):
    """
    Solves conditional program.
    """
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
    X, y, budget, c_bar, title="sim", true_cond_densities=None
):
    X, X_mean, X_std = standardize(X)
    y, y_mean, y_std = standardize(y)

    q_hat = QuantileRegressor(quantile=budget, alpha=0, solver="highs").fit(X, y)
    quantile = q_hat.predict(X) * y_std + y_mean
    transfer = np.maximum(c_bar - quantile, 0)

    optimal_policy = {}
    for i in range(len(X)):
        optimal_policy[i] = [(transfer[i], 1.0)]

    total_cost = policy_cost(optimal_policy, np.ones(len(X)) / len(X))

    if true_cond_densities is not None:
        prob = prob_below_line(
            optimal_policy, c_bar, np.ones(len(X)) / len(X), true_cond_densities
        )
    else:
        prob = budget

    results_file = "results/{}_conditional_program.csv".format(title)

    result = {"total_transfer": total_cost, "prob_below_line": prob}
    write_result(results_file, result)
    return optimal_policy, total_cost
