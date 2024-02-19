from data_loaders.semi_sim_data_gen import get_semi_synthetic_malawi_data
from knapsack import compute_alpha_opt_policies
from conditional_program import (
    solve_conditional_program,
)
from utils import make_density_plot, get_cond_density_estimator, log_likelihood
from reporting import write_result
from evaluate import post_transfer_metrics_true_dist
from data_loaders.data_utils import split_data

import numpy as np
import argh

np.random.seed(123456)


def run_alg(train_dataset, cond_density_estimator, budget, c_bar):
    title = "malawi_synthetic_n={}_d={}".format(
        len(train_dataset), train_dataset.X.shape[1]
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
        n_alpha=200,
        title="{}_estimated_train".format(title),
    )

    idx = np.argmin(train_total_transfers_est)
    t_joint_program_est = t_alpha_joint_programs_est[idx]

    return (
        t_cond_program_est,
        t_joint_program_est,
    )


def run_ground_truth(dataset, cond_density_true, budget, c_bar):
    title = "malawi_synthetic_n={}_d={}".format(len(dataset), dataset.X.shape[1])
    print("Solving conditional program using true cond densities...")
    t_cond_program_true = solve_conditional_program(cond_density_true, budget, c_bar)
    print("Solving joint program using true cond densities...")
    (
        t_alpha_joint_programs_true,
        total_transfers_true,
        alphas,
    ) = compute_alpha_opt_policies(
        dataset,
        cond_density_true,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_true".format(title),
    )
    idx = np.argmin(total_transfers_true)
    t_joint_program_true = t_alpha_joint_programs_true[idx]
    return t_cond_program_true, t_joint_program_true


def evaluate(test_dataset, true_cond_densities, policy, c_bar, title, metadata):
    result = post_transfer_metrics_true_dist(
        test_dataset, true_cond_densities, policy, c_bar
    )
    metadata.update(result)
    results_file = "results/{}.csv".format(title)
    write_result(results_file, metadata)


@argh.arg("--d", default=2)
@argh.arg("--budget", default=0.1)
@argh.arg("--density_est_method", default="glm")
def main(d=2, budget=0.1, density_est_method="glm"):
    n = 20000
    X, y, cond_density_true = get_semi_synthetic_malawi_data(n, d)
    print("d", d, "density_est_method", density_est_method)
    train_dataset, test_dataset = split_data(X, y, r=None, p=0.5)
    c_bar = 2.15
    print("c_bar:{}".format(c_bar))

    cond_density_estimator = get_cond_density_estimator(
        train_dataset, density_est_method, 10
    )

    title = "Malawi Synthetic (n={}, d={})".format(len(train_dataset), d)

    outcome_range = (0.0, np.quantile(y, 0.92))
    make_density_plot(
        train_dataset,
        cond_density_estimator,
        cond_density_true,
        outcome_range,
        title,
    )

    est_avg_log_likelihood = log_likelihood(test_dataset, cond_density_estimator)
    print("Est Average LL: {}".format(est_avg_log_likelihood))

    true_avg_log_likelihood = log_likelihood(test_dataset, cond_density_true)
    print("True Average LL: {}".format(true_avg_log_likelihood))

    t_cond_program_true, t_joint_program_true = run_ground_truth(
        test_dataset, cond_density_true, budget, c_bar
    )

    t_cond_program_est, t_joint_program_est = run_alg(
        test_dataset, cond_density_estimator, budget, c_bar
    )

    title = "malawi_synthetic_n={}".format(len(train_dataset))

    metadata = {
        "density_est_method": "true",
        "d": d,
        "method": "cond_program",
        "budget": budget,
    }

    policies = [
        t_cond_program_est,
        t_cond_program_true,
        t_joint_program_est,
        t_joint_program_true,
    ]
    names = ["cond_program", "cond_program", "joint_program", "joint_program"]
    density_est_methods = ["density", "oracle", "density", "oracle"]

    true_cond_densities = cond_density_true(test_dataset.X)
    for i in range(len(policies)):
        metadata["method"] = names[i]
        metadata["density_est_method"] = density_est_methods[i]
        evaluate(
            test_dataset,
            true_cond_densities,
            policies[i],
            c_bar,
            title=title,
            metadata=metadata,
        )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
