import numpy as np
import matplotlib.pyplot as plt


def make_density_plot(X_train, cond_density_estimator, true_cond_densities, title):
    estimated_cond_densities = cond_density_estimator(X_train)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    subset_cond = true_cond_densities + estimated_cond_densities
    a = np.linspace(0.0, max([1.5 * dist.mode for dist in subset_cond]), 1000)
    for i in range(len(true_cond_densities)):
        ax[0].plot(a, true_cond_densities[i].pdf(a))
        ax[1].plot(a, estimated_cond_densities[i].pdf(a))

    for i in range(2):
        ax[i].set_ylim(0, max([dist.pdf(dist.mode) for dist in subset_cond]) + 0.05)

    ax[0].set_title("True Conditional Densities")
    ax[1].set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()


def make_estimated_density_plot(X_train, cond_density_estimator, title):
    estimated_cond_densities = cond_density_estimator(X_train)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    a = np.linspace(
        0.0, max([1.5 * dist.mode for dist in estimated_cond_densities]), 1000
    )
    for i in range(len(estimated_cond_densities)):
        ax.plot(a, estimated_cond_densities[i].pdf(a))

    ax.set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()
