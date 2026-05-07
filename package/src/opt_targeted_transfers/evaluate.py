import numpy as np


def expected_value_transfers(assignments):
    transfers = []
    for i in range(len(assignments)):
        ev = 0.0
        for j in range(len(assignments[i])):
            ev += assignments[i][j][1] * assignments[i][j][0]

        transfers.append(ev)
    return np.array(transfers)


def policy_cost(test_covariate_dataset, assignments):
    _, r_test = test_covariate_dataset.get_data()
    cost = 0.0
    for i in range(len(assignments)):
        for j in range(len(assignments[i])):
            cost += r_test[i] * assignments[i][j][1] * assignments[i][j][0]
    return cost


def post_transfer_metrics(test_dataset, assignments, c_bar):
    """
    Compute post-transfer metrics for a policy given the test dataset.

    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    :param policy: The policy function.
    :type policy: Callable[[np.ndarray], np.ndarray]
    :param c_bar: The minimum threshold value.
    :type c_bar: float
    :param oracle: Whether or not policy is the oracle policy.
    :type oracle: bool
    :return: A dictionary of post-transfer metrics.
    :rtype: dict
    """

    _, y_test, r_test = test_dataset.get_data()

    dic = {
        "initial_poverty_rate": 0.0,
        "initial_poverty_gap": 0.0,
        "initial_welfare": 0.0,
        "post_transfer_poverty_gap": 0.0,
        "post_transfer_poverty_rate": 0.0,
        "post_transfer_welfare": 0.0,
        "policy_cost_per_capita": 0.0,
    }

    dic["initial_poverty_gap"] = np.sum(np.maximum(c_bar - y_test, 0) * r_test).item()
    dic["initial_poverty_rate"] = np.sum(r_test * (y_test < c_bar))
    dic["initial_welfare"] = np.sum(np.log(np.clip(y_test, a_min=1e-5, a_max=None)) * r_test).item()

    for i in assignments:
        pov_gap = 0.0
        pov_rate = 0.0
        cost = 0.0
        welfare = 0.0
        for j in range(len(assignments[i])):
            transfer_amt = assignments[i][j][0]
            prob = assignments[i][j][1]

            pov_gap += prob * np.maximum(c_bar - y_test[i] - transfer_amt, 0)
            pov_rate += prob * (transfer_amt + y_test[i] < c_bar).astype(float)
            cost += prob * transfer_amt
            welfare += prob * np.log(y_test[i] + transfer_amt)

        dic["post_transfer_poverty_gap"] += pov_gap * r_test[i]
        dic["post_transfer_poverty_rate"] += pov_rate * r_test[i]
        dic["post_transfer_welfare"] += welfare * r_test[i]
        dic["policy_cost_per_capita"] += cost * r_test[i]

    return dic
