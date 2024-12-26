from gan_data_gen import generate_synthetic_data
import dill
import pandas as pd
import numpy as np


def get_wgan_data_generator(generatorpath, datawrapperpath):
    """
    Load a trained WGAN generator and data wrapper from disk and return a data generator function.

    Args:
        generatorpath (str): Path to the trained WGAN generator.
        datawrapperpath (str): Path to the data wrapper for the WGAN generator.

    Returns:
        data_generator (function): A function that generates synthetic data using the trained WGAN generator
    """
    with open(generatorpath, "rb") as dill_file:
        generator = dill.load(dill_file)

    with open(datawrapperpath, "rb") as dill_file:
        data_wrapper = dill.load(dill_file)

    def data_generator(nsamples, seed):
        return generate_synthetic_data(generator, data_wrapper, nsamples, seed)

    return data_generator


def get_gt_data_generator(trainpath):
    """
    Load the ground truth training dataset and return a data generator function.

    Args:
        trainpath (str): Path to the ground truth training dataset.

    Returns:
        data_generator (function): A function that generates samples from the ground truth dataset.
    """

    data = pd.read_parquet(trainpath)

    # some of this preprocessing code should eventually be deprecated because
    # it should be handled by prior data preprocessing code

    # compute outcome conversion factor
    a = 340.2 / 430.05  # Malawi CPI in 2017 USD / Malawi CPI in 2019 USD
    b = 241.98  # Malawi Kwacha to USD exchange rate in 2017
    adulteq = data["adulteq"]
    # can alternatively implement this as data["num_adults"] + alpha * data["num_children"]
    # where alpha is in (0, 1).
    conversion_factor = (a / b) * (1 / 365) * (1 / adulteq)
    data["consumption_per_capita_per_day"] = data["rexpagg"] * conversion_factor

    # we include hh_wgt and consumption_per_capita_per_day so that
    # we can synthetically generate samples from the joint distribution (X, Y, R)
    durable_verifiable_covariates = list(
        pd.read_csv("data/durable_verifiable_covariates.csv")["Covariates"]
    )

    data = data[
        durable_verifiable_covariates + ["consumption_per_capita_per_day", "hh_wgt"]
    ]

    # # More appropriate to represent this variable on a log scale
    # data["log_yearly_rent"] = np.log1p(data["yearly_rent"])
    # del data["yearly_rent"]

    def data_generator(nsamples, seed):
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(data.index, nsamples, replace=True)
        return data.loc[sample_indices].reset_index(drop=True)

    return data_generator
