from knapsack import compute_alpha_opt_policies
from conditional_program import (
    solve_conditional_program_quantile_regression,
    solve_conditional_program,
)
from package.src.opt_targeted_transfers.oracle import run_oracle
from evaluate import post_transfer_metrics
from utils import (
    make_estimated_density_plot,
    get_cond_density_estimator,
    log_likelihood,
)
from data_loaders.data_utils import split_data
from data_loaders.data_loader import load_dataset
from reporting import write_result


import numpy as np
import argh
import dill as pickle


def run_cond_alg(train_dataset, cond_density_estimator, budget, c_bar):
    t_cond_program_qr = solve_conditional_program_quantile_regression(
        train_dataset, budget, c_bar
    )

    t_cond_program_est = solve_conditional_program(
        cond_density_estimator, budget, c_bar
    )
    return t_cond_program_qr, t_cond_program_est


def run_main_alg(dataset, cond_density_estimator, budget, c_bar, country):
    title = "{}_d={}_budget={}".format(country, dataset.X.shape[1], budget)

    (
        t_alpha_joint_programs,
        train_total_transfers,
        alphas,
    ) = compute_alpha_opt_policies(
        dataset,
        cond_density_estimator,
        budget,
        c_bar,
        n_alpha=200,
        title="{}_joint_opt".format(title),
    )

    idx = np.argmin(train_total_transfers)
    t_joint_program_est = t_alpha_joint_programs[idx]
    import pdb

    pdb.set_trace()
    return t_joint_program_est


def evaluate(test_dataset, policy, c_bar, title, metadata):
    if metadata["method"] == "oracle":
        result = post_transfer_metrics(test_dataset, policy, c_bar, oracle=True)
    else:
        result = post_transfer_metrics(test_dataset, policy, c_bar)

    metadata.update(result)
    results_file = "results/{}.csv".format(title)
    write_result(results_file, metadata)


@argh.arg("--budget", default=0.1)
@argh.arg("--d", default=2)
@argh.arg("--density_est_method", default="glm")
@argh.arg("--country", default="uganda")
def main(country="uganda", d=2, budget=0.1, density_est_method="glm"):
    X, y, r, features = load_dataset(country)
    # dont use sample weights until we fix knapsack algorithm
    #    trunc_range = (min(y), np.quantile(y, 0.99))
    train_dataset, test_dataset = split_data(X[:, :d], y, r=None, p=0.6)

    max_d = X.shape[1]
    n = len(train_dataset)
    c_bar = 2.15
    print("c_bar:{}".format(c_bar), "budget:{}".format(budget))
    print("Features: ", features[:d])
    cond_density_estimator = get_cond_density_estimator(
        train_dataset, density_est_method
    )

    pickle.dump(
        cond_density_estimator,
        open("{}_cond_density_estimator_d={}.pickle".format(country, d), "wb"),
    )

    make_estimated_density_plot(
        train_dataset,
        cond_density_estimator,
        outcome_range=(min(y), np.quantile(y, 0.99)),
        title="{}_n={}_d={}".format(country, n, d),
    )
    train_avg_log_likelihood = log_likelihood(
        train_dataset, cond_density_estimator, full_X=False
    )
    est_avg_log_likelihood = log_likelihood(
        test_dataset, cond_density_estimator, full_X=False
    )
    print(
        "Train Average LL: {}, Test Average LL: {}".format(
            train_avg_log_likelihood, est_avg_log_likelihood
        )
    )

    t_cond_program_qr, t_cond_program_est = run_cond_alg(
        train_dataset, cond_density_estimator, budget, c_bar
    )

    metadata = {
        "density_est_method": density_est_method,
        "d": d,
        "budget": budget,
        "avg_log_likelihood_density": est_avg_log_likelihood,
    }

    t_oracle = run_oracle(test_dataset, budget, c_bar)

    policies = [t_cond_program_qr, t_cond_program_est, t_oracle]
    names = ["cond_program_qr", "cond_program_density", "oracle"]

    for i in range(len(policies)):
        metadata["method"] = names[i]
        evaluate(
            test_dataset,
            policies[i],
            c_bar,
            title="{}_n={}".format(country, n),
            metadata=metadata,
        )

    t_joint_program_est = run_main_alg(
        test_dataset, cond_density_estimator, budget, c_bar, country
    )
    metadata["method"] = "joint_program"
    evaluate(
        test_dataset,
        t_joint_program_est,
        c_bar,
        title="{}_n={}".format(country, n),
        metadata=metadata,
    )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
