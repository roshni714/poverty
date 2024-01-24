from data_loaders.sim_data_gen import generate_heteroscedastic_data
from mle import get_cond_density_estimator
from knapsack import compute_alpha_opt_policies
from conditional_program import (
    solve_conditional_program_quantile_regression,
    solve_conditional_program,
)
from utils import make_density_plot
from reporting import write_result
from evaluate import post_transfer_metrics
from data_loaders.data_utils import split_data

import numpy as np
import argh


def run_alg(train_dataset, cond_density_estimator, budget, c_bar):
    title = "heteroscedastic_n={}_d={}".format(
        len(train_dataset), train_dataset.X.shape[1]
    )
    print("Solving cond program using QR...")
    t_cond_program_qr = solve_conditional_program_quantile_regression(
        train_dataset, budget, c_bar
    )

    print("Solving cond program using estimated cond densities...")
    t_cond_program_est = solve_conditional_program(
        cond_density_estimator, budget, c_bar
    )

    print("Solving joint program using estimated cond densities...")
    (
        t_alpha_joint_programs_est,
        train_total_transfers_est,
        alphas,
    ) = compute_alpha_opt_policies(
        train_dataset,
        cond_density_estimator,
        budget,
        c_bar,
        n_alpha=10,
        title="{}_estimated_train".format(title),
    )

    idx = np.argmin(train_total_transfers_est)
    t_joint_program_est = t_alpha_joint_programs_est[idx]

    return (
        t_cond_program_qr,
        t_cond_program_est,
        t_joint_program_est,
    )


def run_ground_truth(train_dataset, cond_density_true, budget, c_bar):
    title = "heteroscedastic_n={}_d={}".format(
        len(train_dataset), train_dataset.full_X.shape[1]
    )
    print("Solving conditional program using true cond densities...")
    t_cond_program_true = solve_conditional_program(cond_density_true, budget, c_bar)
    print("Solving joint program using true cond densities...")
    (
        t_alpha_joint_programs_true,
        train_total_transfers_true,
        alphas,
    ) = compute_alpha_opt_policies(
        train_dataset,
        cond_density_true,
        budget,
        c_bar,
        n_alpha=10,
        title="{}_true_train".format(title),
        full_X=True,
    )
    idx = np.argmin(train_total_transfers_true)
    t_joint_program_true = t_alpha_joint_programs_true[idx]
    return t_cond_program_true, t_joint_program_true


def evaluate(test_dataset, policy, c_bar, title, full_X):
    result = post_transfer_metrics(test_dataset, policy, c_bar, full_X=full_X)
    results_file = "results/{}.csv".format(title)
    write_result(results_file, result)


@argh.arg("--d", default=2)
def main(d=2):
    n = 20000
    max_d = 20
    X, y, cond_density_true = generate_heteroscedastic_data(n, max_d)
    train_dataset, test_dataset = split_data(X, y, r=None, d=d, p=0.25)

    budget = 0.1
    c_bar = np.quantile(y, budget * 2)
    print("c_bar:{}".format(c_bar))
    cond_density_estimator = get_cond_density_estimator(train_dataset)
    title = "heteroscedastic_n={}_d={}".format(
        len(train_dataset), train_dataset.X.shape[1]
    )

    make_density_plot(
        train_dataset,
        cond_density_estimator,
        cond_density_true,
        title,
    )

    t_cond_program_true, t_joint_program_true = run_ground_truth(
        train_dataset, cond_density_true, budget, c_bar
    )
    t_cond_program_qr, t_cond_program_est, t_joint_program_est = run_alg(
        train_dataset, cond_density_estimator, budget, c_bar
    )

    evaluate(
        test_dataset,
        t_cond_program_true,
        c_bar,
        title="heteroscedastic_n={}_d={}_cond_program_true".format(n, max_d),
        full_X=True,
    )
    evaluate(
        test_dataset,
        t_joint_program_true,
        c_bar,
        title="heteroscedastic_n={}_d={}_joint_program_true".format(n, max_d),
        full_X=True,
    )

    evaluate(
        test_dataset,
        t_cond_program_qr,
        c_bar,
        title="heteroscedastic_n={}_d={}_cond_program_qr".format(n, d),
        full_X=False,
    )
    evaluate(
        test_dataset,
        t_cond_program_est,
        c_bar,
        title="heteroscedastic_n={}_d={}_cond_program_est".format(n, d),
        full_X=False,
    )
    evaluate(
        test_dataset,
        t_joint_program_est,
        c_bar,
        title="heteroscedastic_n={}_d={}_joint_program_est".format(n, d),
        full_X=False,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
