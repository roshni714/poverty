import numpy as np


def expected_value_transfers(test_dataset, policy):
    y_test = test_dataset.y
    assignments = policy(test_dataset.X)

    transfers = []
    for i in range(len(test_dataset)):
        ev = 0.0
        for j in range(len(assignments[i])):
            ev += assignments[i][j][1] * assignments[i][j][0]

        transfers.append(ev)
    return np.array(transfers)


def post_transfer_metrics(test_dataset, policy, c_bar, oracle=False):
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
    dic = {
        "initial_poverty_rate": 0.0,
        "initial_poverty_gap": 0.0,
        "post_transfer_poverty_gap": 0.0,
        "post_transfer_poverty_rate": 0.0,
        "policy_cost": 0.0,
    }

    y_test = test_dataset.y
    r_test = test_dataset.r

    dic["initial_poverty_gap"] = np.sum(np.maximum(c_bar - y_test, 0) * r_test).item()
    dic["initial_poverty_rate"] = np.sum(r_test * (y_test < c_bar))

    if not oracle:
        assignments = policy(test_dataset.X)
    else:
        assignments = policy(test_dataset.y)

    for i in range(len(test_dataset)):
        pov_gap = 0.0
        prob = 0.0
        cost = 0.0
        for j in range(len(assignments[i])):
            pov_gap += assignments[i][j][1] * np.maximum(
                c_bar - y_test[i] - assignments[i][j][0], 0
            )
            prob += assignments[i][j][1] * (
                assignments[i][j][0] + y_test[i] < c_bar
            ).astype(float)
            cost += assignments[i][j][1] * assignments[i][j][0]

        dic["post_transfer_poverty_gap"] += pov_gap * r_test[i]
        dic["post_transfer_poverty_rate"] += prob * r_test[i]
        dic["policy_cost"] += cost * r_test[i]

    return dic
