import numpy as np
import matplotlib.pyplot as plt


def make_density_plot(train_dataset, cond_density_estimator, cond_density_true, title):
    estimated_cond_densities = cond_density_estimator(train_dataset.X[:10, :])
    true_cond_densities = cond_density_true(train_dataset.full_X[:10, :])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    subset_cond = list(true_cond_densities) + list(estimated_cond_densities)
    a = np.linspace(0.0, max([1.5 * dist.mode for dist in subset_cond]), 1000)
    for i in range(10):
        ax[0].plot(a, true_cond_densities[i].pdf(a))
        ax[1].plot(a, estimated_cond_densities[i].pdf(a))

    for i in range(2):
        ax[i].set_ylim(0, max([dist.pdf(dist.mode) for dist in subset_cond]) + 0.05)

    ax[0].set_title("True Conditional Densities")
    ax[1].set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()


def make_estimated_density_plot(train_dataset, cond_density_estimator, title):
    estimated_cond_densities = cond_density_estimator(train_dataset.X[:10, :])
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    a = np.linspace(
        0.0, max([1.5 * dist.mode for dist in estimated_cond_densities]), 1000
    )
    for i in range(10):
        ax.plot(a, estimated_cond_densities[i].pdf(a))

    ax.set_title("Estimated Conditional Densities")
    plt.suptitle("{} Conditional Distributions".format(title))
    plt.savefig("figs/{}.pdf".format(title))
    plt.show()
