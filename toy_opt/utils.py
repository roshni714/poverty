import numpy as np
import matplotlib.pyplot as plt
from glm import get_glm_spline_fit_helper
from mle import get_lognormal_fit_helper


def log_likelihood(test_dataset, cond_density_estimator, full_X=False):
    y_test = test_dataset.y

    if full_X:
        X_test = test_dataset.full_X
    else:
        X_test = test_dataset.X

    r = test_dataset.r
    log_likelihood = 0.0
    cond_dists = cond_density_estimator(X_test)

    y_test = np.clip(
        y_test,
        cond_dists[0].outcome_range[0] + 1e-5,
        cond_dists[0].outcome_range[1] - 1e-5,
    )
    log_likelihoods = [
        np.log(cond_dists[i].pdf(y_test[[i]]) + 1e-10) * r[i]
        for i in range(len(y_test))
    ]
    return np.sum(log_likelihoods)


def get_cond_density_estimator(train_dataset, budget, method):
    if method == "log_normal":
        cond_density_estimator = get_lognormal_fit_helper(train_dataset)

    elif method == "glm_spline":
        cond_density_estimator = get_glm_spline_fit_helper(train_dataset, budget)

    return cond_density_estimator


def make_density_plot(
    train_dataset, cond_density_estimator, cond_density_true, outcome_range, title
):
    n_dist = 20
    estimated_cond_densities = cond_density_estimator(train_dataset.X[:n_dist, :])
    true_cond_densities = cond_density_true(train_dataset.full_X[:n_dist, :])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    subset_cond = list(true_cond_densities) + list(estimated_cond_densities)
    a = np.linspace(outcome_range[0], outcome_range[1], 1000)
    for i in range(n_dist):
        ax[0].plot(a, true_cond_densities[i].pdf(a))
        ax[1].plot(a, estimated_cond_densities[i].pdf(a))

    ax[0].set_title("True Conditional Densities")
    ax[1].set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()


def make_estimated_density_plot(
    train_dataset, cond_density_estimator, outcome_range, title
):
    n_dist = 20
    estimated_cond_densities = cond_density_estimator(train_dataset.X[:n_dist, :])
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    a = np.linspace(outcome_range[0], outcome_range[1], 1000)
    for i in range(n_dist):
        ax.plot(a, estimated_cond_densities[i].pdf(a))

    ax.set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()
