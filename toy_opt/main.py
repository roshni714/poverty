from data_loaders.sim_data_gen import generate_heteroscedastic_data
from knapsack import compute_alpha_opt_policies
from conditional_program import (
    solve_conditional_program_quantile_regression,
    solve_conditional_program,
)
from utils import make_density_plot, get_cond_density_estimator, log_likelihood
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


def evaluate(test_dataset, policy, c_bar, title, full_X, metadata):
    result = post_transfer_metrics(test_dataset, policy, c_bar, full_X=full_X)
    metadata.update(result)
    results_file = "results/{}.csv".format(title)
    write_result(results_file, metadata)


@argh.arg("--d", default=2)
@argh.arg("--density_est_method", default="log_normal")
def main(d=2, density_est_method="log_normal"):
    n = 20000
    max_d = 10
    X, y, cond_density_true = generate_heteroscedastic_data(n, max_d)
    print("d", d, "density_est_method", density_est_method)
    outcome_range = (0.0, np.quantile(y, 0.98))
    train_dataset, test_dataset = split_data(
        X, y, r=None, d=d, p=0.5, outcome_range=outcome_range
    )

    budget = 0.1
    true_densities = cond_density_true(train_dataset.full_X)
    c_bar = np.mean([density.ppf(budget * 2) for density in true_densities])
    print("c_bar:{}".format(c_bar))
    cond_density_estimator = get_cond_density_estimator(
        train_dataset, density_est_method, outcome_range
    )
    title = "heteroscedastic_n={}_d={}_{}".format(
        len(train_dataset), d, density_est_method
    )

    make_density_plot(
        train_dataset,
        cond_density_estimator,
        cond_density_true,
        outcome_range,
        title,
    )

    est_avg_log_likelihood = log_likelihood(
        test_dataset, cond_density_estimator, outcome_range
    )
    print("Average LL: {}".format(est_avg_log_likelihood))
    true_avg_log_likelihood = log_likelihood(
        test_dataset, cond_density_true, outcome_range
    )
    print("True Average LL: {}".format(true_avg_log_likelihood))

    title = "heteroscedastic_n={}".format(len(train_dataset))

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
        title=title,
        full_X=True,
        metadata={
            "density_est_method": "true",
            "d": max_d,
            "avg_log_likelihood_density": true_avg_log_likelihood,
            "method": "cond_program_exact",
        },
    )
    evaluate(
        test_dataset,
        t_joint_program_true,
        c_bar,
        title=title,
        full_X=True,
        metadata={
            "density_est_method": "true",
            "d": max_d,
            "avg_log_likelihood_density": true_avg_log_likelihood,
            "method": "joint_program",
        },
    )

    metadata = {
        "density_est_method": density_est_method,
        "d": d,
        "avg_log_likelihood_density": est_avg_log_likelihood,
    }

    policies = [t_cond_program_qr, t_cond_program_est, t_joint_program_est]
    names = ["cond_program_qr", "cond_program_exact", "joint_program"]

    for i in range(len(policies)):
        metadata["method"] = names[i]
        evaluate(
            test_dataset,
            policies[i],
            c_bar,
            title=title,
            full_X=False,
            metadata=metadata,
        )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
