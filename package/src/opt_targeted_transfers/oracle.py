from opt_targeted_transfers.knapsack import (
    solve_fractional_mc_knapsack_problem,
)
from opt_targeted_transfers.cond_dist import get_lower_cvx_hull

import bisect
import numpy as np

def run_oracle_poverty_rate(test_dataset, budget, c_bar):
    # Behaves slightly wrong if multiple households have identical income and they
    # happen to fall right on the border between those receiving and not receiving
    # transfers. Identical incomes are unlikely given how consumption aggregates are
    # calculated. Will fix eventually.
    _, y_test, r_test = test_dataset.get_data()
    dist_to_line = np.maximum(c_bar - y_test, 0)
    sorting_indices = np.argsort(dist_to_line)
    weighted_transfers = r_test[sorting_indices] * dist_to_line[sorting_indices]
    indicator_receive_transfers = np.cumsum(weighted_transfers) <= budget
    idx_receive_transfers = sorting_indices[indicator_receive_transfers]
    assignments = {i: [(0., 1.0)] for i in range(len(y_test))}
    for idx in idx_receive_transfers:
        assignments[idx] = [(dist_to_line[idx], 1.0)]
    return assignments

def run_oracle_poverty_gap_lift_to_line_scheme(test_dataset, budget, c_bar):
    return run_oracle_poverty_rate(test_dataset, budget, c_bar)


def run_oracle_poverty_gap_floor_scheme(test_dataset, budget, c_bar):

    _, y_test, r_test = test_dataset.get_data()

    gaps = np.maximum(c_bar - y_test, 0)

    sorting_indices = np.argsort(gaps)[::-1] # sort gaps from largest to smallest
    budget_remaining = budget

    running_transfers = 0.

    assignments = {i: [(0., 1.0)] for i in range(len(y_test))}

    for i in sorting_indices:
        gap_contribution = gaps[i] * r_test[i]

        if budget_remaining > gap_contribution:
            budget_remaining -= gap_contribution
            running_transfers += gap_contribution
            assignments[i] = [(gaps[i], 1.0)]
        else:
            prob = budget_remaining / gap_contribution
            assignments[i] = [(gaps[i], prob), (0., 1- prob)]
            break

    return assignments


