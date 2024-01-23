import numpy as np


def empirical_poverty_gap(y, p_xs, c_bar):
    val = np.maximum(c_bar - y, 0)
    return np.sum(val * p_xs)


def poverty_gap(cond_densities, p_xs, c_bar):
    """
    p_xs: (N,)
    """

    def f(z):
        return np.maximum(c_bar - z, 0)

    ev = [density.expect(f) for density in cond_densities]
    return np.sum(ev * p_xs)


def post_transfer_metrics_empirical(test_dataset, policy, budget, c_bar):
    dic = {
        "initial_poverty_gap": 0.0,
        "post_transfer_poverty_gap": 0.0,
        "prob_below_line": 0.0,
        "policy_cost": 0.0,
    }

    X_test = test_dataset.X
    y_test = test_dataset.y
    r_test = test_dataset.r
    assignments = policy(X_test, r_test)

    dic["initial_poverty_gap"] = np.sum(np.maximum(c_bar - y_test, 0) * r_test).item()

    for i in range(len(test_dataset.X)):
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


def post_transfer_metrics(cond_densities, p_xs, assignments, c_bar):
    def f(t):
        def f2(z):
            return np.maximum(c_bar - z - t, 0)

        return f2

    dic = {"post_transfer_poverty_gap": 0.0, "prob_below_line": 0.0, "policy_cost": 0.0}
    for i in range(len(p_xs)):
        pov_gap = 0.0
        prob = 0.0
        cost = 0.0
        for j in range(len(assignments[i])):
            pov_gap += assignments[i][j][1] * cond_densities[i].expect(
                f(assignments[i][j][0])
            )
            prob += (
                cond_densities[i].cdf(c_bar - np.maximum(assignments[i][j][0], 0))
                * assignments[i][j][1]
            )
            cost += assignments[i][j][0] * assignments[i][j][1]

        dic["post_transfer_poverty_gap"] += pov_gap * p_xs[i]
        dic["prob_below_line"] += prob * p_xs[i]
        dic["policy_cost"] += cost * p_xs[i]

    return dic
