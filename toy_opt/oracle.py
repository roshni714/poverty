from knapsack import solve_fractional_knapsack_problem, check_assignments_are_equal
from cond_dist import get_lower_cvx_hull

import bisect
import numpy as np


def get_transfer_function(c_bar, eta, lamb):
    def t(X_test, y_test):
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        transfers = np.maximum(c_bar - y_test, 0)
        below_line = y_test < c_bar

        for j in range(len(X_test)):
            cvx_hull = get_lower_cvx_hull(
                [(0, c_bar), (0, transfers[j]), (below_line[j], 0.0)]
            )
            ratios = np.zeros(len(cvx_hull)).astype(np.float64)
            ratios[0] = -np.inf
            for i in range(len(cvx_hull) - 1):
                p1 = cvx_hull[i]
                p2 = cvx_hull[i + 1]
                ratios[i + 1] = (p2[1] - p1[1]) / (p2[0] - p1[0])
            idx = bisect.bisect_left(ratios, eta)

            if (
                idx > 0
                and idx < len(ratios)
                and ratios[idx - 1] < eta
                and ratios[idx] > eta
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == eta:
                assignments[j] = [
                    (cvx_hull[idx - 1][1], lamb),
                    (cvx_hull[idx][1], 1 - lamb),
                ]
            else:
                assignments[j] = [(0.0, 1.0)]
        return assignments

    return t


def run_oracle(dataset, budget, c_bar):
    X = dataset.X
    y = dataset.y
    r = dataset.r
    transfers = np.maximum(c_bar - y, 0)
    below_line = y < c_bar

    convex_hulls = [
        get_lower_cvx_hull([(0.0, c_bar), (0.0, transfers[i]), (below_line[i], 0.0)])
        for i in range(len(dataset))
    ]

    (
        opt_assignment,
        total_transfer,
        prob_below_line,
        eta,
        lamb,
    ) = solve_fractional_knapsack_problem(r, convex_hulls, budget)
    oracle_policy = get_transfer_function(c_bar, eta, lamb)

    prox_assignment = oracle_policy(X, y)
    check_assignments_are_equal(opt_assignment, prox_assignment)

    return oracle_policy
