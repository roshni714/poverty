import numpy as np


def post_transfer_metrics(test_dataset, policy, c_bar, oracle=False):
    dic = {
        "initial_prob_below_line": 0.0,
        "initial_poverty_gap": 0.0,
        "post_transfer_poverty_gap": 0.0,
        "post_transfer_prob_below_line": 0.0,
        "policy_cost": 0.0,
    }

    y_test = test_dataset.y
    r_test = test_dataset.r

    dic["initial_poverty_gap"] = np.sum(np.maximum(c_bar - y_test, 0) * r_test).item()
    dic["initial_prob_below_line"] = np.sum(r_test * (y_test < c_bar))

    if not oracle:
        assignments = policy(test_dataset.X)
    else:
        assignments = policy(test_dataset.X, test_dataset.y)

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
        dic["post_transfer_prob_below_line"] += prob * r_test[i]
        dic["policy_cost"] += cost * r_test[i]

    return dic


def post_transfer_metrics_true_dist(dataset, cond_densities, policy, c_bar):
    dic = {
        "initial_prob_below_line": 0.0,
        "post_transfer_prob_below_line": 0.0,
        "policy_cost": 0.0,
    }

    r = dataset.r
    assignments = policy(dataset.X)

    init_pov_rate = np.array([density.cdf(c_bar) for density in cond_densities])
    init_pov_rate = np.sum(init_pov_rate * r).item()
    dic["initial_prob_below_line"] = init_pov_rate

    for i in range(len(dataset)):
        prob = 0.0
        cost = 0.0
        density = cond_densities[i]
        for j in range(len(assignments[i])):
            prob += assignments[i][j][1] * density.cdf(c_bar - assignments[i][j][0])
            cost += assignments[i][j][1] * assignments[i][j][0]

        dic["post_transfer_prob_below_line"] += prob * r[i]
        dic["policy_cost"] += cost * r[i]

    return dic
