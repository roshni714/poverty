from sim_data_gen import generate_homoscedastic_data, generate_heteroscedastic_data
from mle import get_estimated_cond_densities
from knapsack import compute_alpha_opt_policies
from conditional_program import solve_conditional_program_quantile_regression
from utils import make_plot

import numpy as np


def main(n, d):
    X, y, true_cond_densities, gamma, gamma0, psi, psi0 = generate_heteroscedastic_data(
        n, d
    )
    title = "heteroscedastic_n={}_d={}".format(n, d)
    budget = 0.1
    c_bar = 8.0
    p_xs = np.ones(n) / n
    (
        opt_policy_conditional_program,
        total_cost,
    ) = solve_conditional_program_quantile_regression(
        X, y, budget, c_bar, title=title, true_cond_densities=true_cond_densities
    )

    estimated_cond_densities = get_estimated_cond_densities(X, y)

    make_plot(true_cond_densities, estimated_cond_densities, title)

    opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
        estimated_cond_densities,
        p_xs,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_estimated".format(title),
        true_cond_densities=true_cond_densities,
    )
    opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
        true_cond_densities,
        p_xs,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_true".format(title),
    )


n = 5000
ds = [2, 3, 5, 8]

for d in ds:
    print("running d = {}".format(d))
    main(n, d)
