from opt_targeted_transfers.knapsack import (
    solve_fractional_mc_knapsack_problem,
)
from opt_targeted_transfers.cond_dist import get_lower_cvx_hull

import bisect
import numpy as np


def get_transfer_function(c_bar, eta, lamb):
    def t(y_test):
        assignments = {x_idx: [] for x_idx in range(len(y_test))}
        transfers = np.maximum(c_bar - y_test, 0)
        below_line = y_test < c_bar

        for j in range(len(y_test)):
            cvx_hull = get_lower_cvx_hull([(0.0, transfers[j]), (below_line[j], 0.0)])
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
                if lamb != 0:
                    assignments[j] = [
                        (cvx_hull[idx - 1][1], lamb),
                        (cvx_hull[idx][1], 1 - lamb),
                    ]
                else:
                    assignments[j] = [(cvx_hull[idx][1], 1 - lamb)]
            else:
                assignments[j] = [(0.0, 1.0)]
        return assignments

    return t


def run_oracle_poverty_rate(dataset, tolerance, c_bar):
    y = dataset.y
    r = dataset.r
    transfers = np.maximum(c_bar - y, 0)
    below_line = y < c_bar

    convex_hulls = [
        get_lower_cvx_hull([(0.0, transfers[i]), (below_line[i], 0.0)])
        for i in range(len(dataset))
    ]

    (
        opt_assignment,
        total_transfer,
        prob_below_line,
        eta,
        lamb,
    ) = solve_fractional_mc_knapsack_problem(r, convex_hulls, tolerance)
    oracle_policy = get_transfer_function(c_bar, eta, lamb)

    prox_assignment = oracle_policy(y)
    #check_assignments_are_equal(opt_assignment, prox_assignment)

    return oracle_policy


def run_oracle_poverty_gap_lift_to_line_scheme(dataset, tolerance, c_bar):
    # Behaves slightly wrong if multiple households have identical income and they
    # happen to fall right on the border between those receiving and not receiving
    # transfers. Identical incomes are unlikely given how consumption aggregates are
    # calculated. Will fix eventually.

    gaps = np.maximum(c_bar - np.array(dataset.y), 0)

    r = dataset.r
    assert r.sum() == 1

    sorting_indices = np.argsort(gaps)

    tolerance_remaining = tolerance

    wealth_receiving_partial_transfer = 0

    for i in sorting_indices:

        if gaps[i] == 0:
            continue

        gap_contribution = gaps[i] * r[i]

        if tolerance_remaining > gap_contribution:
            tolerance_remaining -= gap_contribution

        else:
            wealth_receiving_partial_transfer = dataset.y[i]
            partial_transfer = gaps[i] - tolerance_remaining / r[i]
            break

    def t(y_test):

        assignments = dict()
        for i in range(len(y_test)):
            if y_test[i] == wealth_receiving_partial_transfer:
                assignments[i] = [(partial_transfer, 1.0)]
            elif y_test[i] < wealth_receiving_partial_transfer:
                assignments[i] = [(c_bar - y_test[i], 1.0)]
            else:
                assignments[i] = [(0.0, 1.0)]
        return assignments

    return t


def run_oracle_poverty_gap_floor_scheme(dataset, tolerance, c_bar):

    gaps = np.maximum(c_bar - np.array(dataset.y), 0)
    r = dataset.r

    assert r.sum() == 1

    sorting_indices = np.argsort(gaps)
    tolerance_remaining = tolerance

    running_weight = 0
    max_poverty_gap = 0

    for i in sorting_indices:
        if gaps[i] > 0:
            running_weight += r[i]

    for i in sorting_indices:

        if gaps[i] == 0:
            continue

        gap_contribution = gaps[i] * running_weight

        if tolerance_remaining > gap_contribution:
            max_poverty_gap = gaps[i]
            tolerance_remaining -= gap_contribution
            running_weight -= r[i]

        else:
            max_poverty_gap += tolerance_remaining / running_weight
            break

    def t(y_test):

        gaps_possibly_negative = c_bar - np.array(y_test)
        transfers = np.maximum(gaps_possibly_negative - max_poverty_gap, 0)
        assignments = {i: [(transfer, 1.0)] for i, transfer in enumerate(transfers)}
        return assignments

    return t
