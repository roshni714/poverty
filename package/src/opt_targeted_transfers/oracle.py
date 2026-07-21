from opt_targeted_transfers.knapsack import (
    solve_fractional_mc_knapsack_problem,
)
from opt_targeted_transfers.cond_dist import get_lower_cvx_hull

import bisect
import numpy as np

def run_oracle_poverty_rate_weakly_equitable(test_dataset, budget, c_bar):
    _, y_test, r_test = test_dataset.get_data()

    upper = c_bar
    lower = 0
    lamb = c_bar /2 

    oracle_gap = np.sum(np.maximum(c_bar - y_test, 0) * r_test)

    if budget >= oracle_gap:
        assignments = {i: [(np.maximum(c_bar - y_test[i], 0), 1.0)] for i in range(len(y_test))}
        return assignments

    def policy_cost(lamb):
        poor = y_test < lamb
        rich = y_test >= lamb
        transfers_to_poor = np.maximum(c_bar - lamb, 0) * poor
        transfers_to_rich = np.maximum(c_bar - y_test, 0) * rich
        transfers = transfers_to_poor + transfers_to_rich
        return np.sum(transfers * r_test), transfers

    cost, transfers = policy_cost(lamb)
    iter = 0
    while np.abs(cost - budget) > 1e-3:
        print(f"Iteration {iter}: cost = {cost}, budget = {budget}, lamb = {lamb}")
        if cost > budget:
            lower = lamb
        else:
            upper = lamb
        lamb = (upper + lower) / 2
        cost, transfers = policy_cost(lamb)
        iter += 1

    assignments = {i: [(transfers[i], 1.0)] for i in range(len(y_test))}
    return assignments
    
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
    assignments = {i: [(0.0, 1.0)] for i in range(len(y_test))}
    for idx in idx_receive_transfers:
        assignments[idx] = [(dist_to_line[idx], 1.0)]
    return assignments


def run_oracle_poverty_gap_lift_to_line_scheme(test_dataset, budget, c_bar):
    return run_oracle_poverty_rate(test_dataset, budget, c_bar)


def run_oracle_poverty_gap_floor_scheme(test_dataset, budget, c_bar):

    _, y_test, r_test = test_dataset.get_data()

    def policy_cost(consumption_floor):
        return np.sum(np.maximum(consumption_floor - y_test, 0) * r_test)

    consumption_floors = np.linspace(0, c_bar, 500)
    i = 0

    while i < len(consumption_floors) and policy_cost(consumption_floors[i]) <= budget:
        i += 1

    consumption_floor = consumption_floors[i - 1]
    assignments = {
        i: [(np.maximum(consumption_floor - y_test[i], 0), 1.0)]
        for i in range(len(y_test))
    }

    return assignments
