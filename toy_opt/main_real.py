from mle import get_estimated_cond_densities
from knapsack import compute_alpha_opt_policies
from conditional_program import solve_conditional_program_quantile_regression
from utils import make_estimated_density_plot
from data_loaders.data_utils import split_data
from data_loaders.data_loader import load_uganda
import numpy as np

import argh


def run_alg(train_dataset, budget, c_bar):
    title = "uganda_n={}_d={}".format(len(train_dataset), train_dataset.X.shape[1])

    opt_policy_conditional_program = solve_conditional_program_quantile_regression(
        train_dataset, budget, c_bar, title=title
    )

    (
        estimated_cond_densities,
        compute_cond_density_helper,
    ) = get_estimated_cond_densities(train_dataset)

    make_estimated_density_plot(estimated_cond_densities, title)
    opt_policies, total_transfers, alphas = compute_alpha_opt_policies(
        estimated_cond_densities,
        train_dataset.r,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_estimated".format(title),
    )

    idx = np.argmin(total_transfers)
    opt_policy_joint_program = opt_policies[idx]
    return opt_policy_conditional_program, opt_policy_joint_program


@argh.arg("--d", default=2)
def main(d=2):
    X, y, r, features = load_uganda()
    train_dataset, test_dataset = split_data(X, y, r=r, d=d, p=0.6)
    n, max_d = X.shape
    budget = 0.1
    c_bar = np.quantile(y, budget * 2)
    print("c_bar:{}".format(c_bar))

    opt_policy_conditional_program, opt_policy_joint_program = run_alg(
        train_dataset, budget, c_bar
    )

    evaluate(
        test_dataset,
        opt_policy_conditional_program,
        title="uganda_n={}_d={}_cond_program_test".format(n, d),
    )
    evaluate(
        test_dataset,
        opt_policy_joint_program,
        title="uganda_n={}_d={}_joint_program_test".format(n, d),
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
