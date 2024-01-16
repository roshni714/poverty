from sim_data_gen import generate_homoscedastic_data, generate_heteroscedastic_data
from mle import get_estimated_cond_densities
from knapsack import compute_alpha_opt_policies
from conditional_program import solve_conditional_program_quantile_regression
from utils import make_plot

import numpy as np


def main(X, y, true_cond_densities, budget, c_bar, d):
    n = len(X)
    title = "heteroscedastic_n={}_d={}".format(n, d)
    p_xs = np.ones(n) / n

    (
        opt_policy_conditional_program,
        total_cost,
    ) = solve_conditional_program_quantile_regression(
        X[:, :d], y, budget, c_bar, title=title, true_cond_densities=true_cond_densities
    )

    estimated_cond_densities = get_estimated_cond_densities(X[:, :d], y)

    #    make_plot(true_cond_densities, estimated_cond_densities, title)
    opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
        estimated_cond_densities,
        p_xs,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_estimated".format(title),
        true_cond_densities=true_cond_densities,
    )


n = 5000
max_d = 10
X, y, true_cond_densities = generate_heteroscedastic_data(n, max_d)
title = "heteroscedastic_n={}_d={}".format(n, max_d)
budget = 0.1
c_bar = np.mean([density.ppf(budget / 2) for density in true_cond_densities])
print("c_bar:{}".format(c_bar))
p_xs = np.ones(n) / n

opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
    true_cond_densities,
    p_xs,
    budget,
    c_bar,
    n_alpha=200,
    title="{}_true".format(title),
)

for d in [2, 3, 5, 8]:
    main(X, y, true_cond_densities, budget, c_bar, d)
