import numpy as np


def post_transfer_metrics(test_dataset, policy, c_bar, full_X=False):
    dic = {
        "initial_poverty_gap": 0.0,
        "post_transfer_poverty_gap": 0.0,
        "prob_below_line": 0.0,
        "policy_cost": 0.0,
    }

    y_test = test_dataset.y
    r_test = test_dataset.r

    if full_X:
        assignments = policy(test_dataset.full_X)
    else:
        assignments = policy(test_dataset.X)

    avg = [assignments[key][0][0] for key in assignments]
    dic["initial_poverty_gap"] = np.sum(np.maximum(c_bar - y_test, 0) * r_test).item()

    for i in range(len(test_dataset)):
        pov_gap = 0.0
        prob = 0.0
        cost = 0.0
        for j in range(len(assignments[i])):
            pov_gap += assignments[i][j][1] * np.maximum(
                c_bar - y_test[i] - assignments[i][j][0], 0
            )
            prob += assignments[i][j][1] * (
                assignments[i][j][0] + y_test[i] <= c_bar
            ).astype(float)
            cost += assignments[i][j][1] * assignments[i][j][0]

        dic["post_transfer_poverty_gap"] += pov_gap * r_test[i]
        dic["prob_below_line"] += prob * r_test[i]
        dic["policy_cost"] += cost * r_test[i]

    return dic
