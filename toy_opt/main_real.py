from knapsack import compute_alpha_opt_policies
from conditional_program import (
    solve_conditional_program_quantile_regression,
    solve_conditional_program,
)
from evaluate import post_transfer_metrics
from utils import make_estimated_density_plot, get_cond_density_estimator
from data_loaders.data_utils import split_data
from data_loaders.data_loader import load_uganda
from reporting import write_result


import numpy as np
import argh


def run_alg(train_dataset, budget, c_bar):
    title = "uganda_n={}_d={}".format(len(train_dataset), train_dataset.X.shape[1])

    t_cond_program_qr = solve_conditional_program_quantile_regression(
        train_dataset, budget, c_bar
    )

    cond_density_estimator = get_cond_density_estimator(train_dataset)

    t_cond_program_est = solve_conditional_program(
        cond_density_estimator, budget, c_bar
    )

    make_estimated_density_plot(train_dataset, cond_density_estimator, title)

    (
        t_alpha_joint_programs,
        train_total_transfers,
        alphas,
    ) = compute_alpha_opt_policies(
        train_dataset,
        cond_density_estimator,
        budget,
        c_bar,
        n_alpha=10,
        title="{}_estimated_train".format(title),
    )

    idx = np.argmin(train_total_transfers)
    t_joint_program_est = t_alpha_joint_programs[idx]
    return t_cond_program_qr, t_cond_program_est, t_joint_program_est


def evaluate(test_dataset, policy, c_bar, title):
    result = post_transfer_metrics(test_dataset, policy, c_bar)
    results_file = "results/{}.csv".format(title)
    write_result(results_file, result)


@argh.arg("--d", default=2)
def main(d=2):
    X, y, r, features = load_uganda()
    # dont use sample weights until we fix knapsack algorithm
    train_dataset, test_dataset = split_data(X, y, r=None, d=d, p=0.5)
    max_d = X.shape[1]
    n = len(train_dataset)
    budget = 0.1
    c_bar = np.quantile(y, budget * 2)
    print("c_bar:{}".format(c_bar))

    t_cond_program_qr, t_cond_program_est, t_joint_program_est = run_alg(
        train_dataset, budget, c_bar
    )

    evaluate(
        test_dataset,
        t_cond_program_qr,
        c_bar,
        title="uganda_n={}_d={}_cond_program_qr".format(n, d),
    )

    evaluate(
        test_dataset,
        t_cond_program_est,
        c_bar,
        title="uganda_n={}_d={}_cond_program_est".format(n, d),
    )

    evaluate(
        test_dataset,
        t_joint_program_est,
        c_bar,
        title="uganda_n={}_d={}_joint_program_est".format(n, d),
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
