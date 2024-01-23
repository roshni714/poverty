from sim_data_gen import generate_homoscedastic_data, generate_heteroscedastic_data
from mle import get_estimated_cond_densities
from knapsack import compute_alpha_opt_policies
from conditional_program import solve_conditional_program_quantile_regression
from utils import make_density_plot
from metrics import poverty_gap, post_transfer_poverty_gap

import numpy as np
import argh


def run_alg(X, y, true_cond_densities, budget, c_bar, d):
    n = len(X)
    title = "heteroscedastic_n={}_d={}".format(n, d)
    p_xs = np.ones(n) / n

    t_cond_program = X[:, :d], y, budget, c_bar, title = (
        title,
        true_cond_densities,
    ) = true_cond_densities

    cond_density_estimator = get_cond_density_estimator(X[:, :d], y)

    make_density_plot(
        train_dataset.X[:10], cond_density_estimator, true_cond_densities[:10], title
    )
    opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
        estimated_cond_densities,
        p_xs,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_estimated".format(title),
        true_cond_densities=true_cond_densities,
    )


@argh.arg("--d", default=2)
def main(d=2):
    n = 5000
    max_d = 20
    X, y, true_cond_densities = generate_heteroscedastic_data(n, max_d)

    X_train, y_train, X_test, y_test = split_data(X, y)
    title = "heteroscedastic_n={}_d={}".format(n, max_d)
    budget = 0.1
    c_bar = np.mean([density.ppf(budget * 2) for density in true_cond_densities])
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

    run_alg(X, y, true_cond_densities, budget, c_bar, d)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
