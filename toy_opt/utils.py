import numpy as np
import matplotlib.pyplot as plt


def standardize(z):
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    data = (z - z_mean) / z_std
    return data, z_mean, z_std


def make_plot(true_cond_densities, estimated_cond_densities, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    subset_cond = true_cond_densities[:10] + estimated_cond_densities[:10]
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
