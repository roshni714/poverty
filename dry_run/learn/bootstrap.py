from learn.aggregation import CountryBootstrapReplicateResults
import numpy as np
from learn.aggregation import CountryMethodPovertyResults
from multiprocessing import Pool


def bootstrap_ci(country, method, metadata, n_bootstrap=100):
    """
    Compute the bootstrap 95% confidence interval width of post-transfer metrics for a given country and method.

    :param country: The country for which to compute the standard error.
    :type country: str
    :param method: The method used for transfer.
    :type method: str
    :param metadata: Metadata containing information about the dataset.
    :type metadata: dict
    :return: The bootstrap 95% confidence interval width of post-transfer metrics.
    :rtype: float
    """
    policy_costs = []
    original = CountryMethodPovertyResults(
        country=country, method=method, metadata=metadata
    )
    mean_policy_cost = original.rate_to_cost_interpolator(metadata.nationalPovertyRate)

    for i in range(n_bootstrap):
        replicate = CountryBootstrapReplicateResults(
            country=country, method=method, metadata=metadata, bootstrap_seed=i
        )
        if replicate.rate_to_cost_interpolator_domain[1] < metadata.nationalPovertyRate:
            policy_costs.append(0.0)
        else:
            policy_cost = replicate.rate_to_cost_interpolator(
                metadata.nationalPovertyRate
            )
            policy_costs.append(policy_cost)
    ci = 1.96 * (
        (np.sum((policy_costs - mean_policy_cost) ** 2) / (n_bootstrap - 1)) ** 0.5
    )
    return ci
