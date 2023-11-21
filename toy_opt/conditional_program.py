import numpy as np


def solve_conditional_program(p_xs, cond_dists, budget, c_bar):
    """
    Solves conditional program.
    """
    cost = []
    assignments = {x_idx: [] for x_idx in range(len(p_xs))}
    for i, cond_dist in enumerate(cond_dists):
        if cond_dist.cdf(c_bar) > budget:
            cost.append((c_bar - cond_dist.ppf(budget)))
            assignments[i] = [c_bar - cond_dist.ppf(budget)]
        else:
            cost.append(0.0)
            assignments[i] = [0.0]

    total_cost = np.sum(np.array(cost) * p_xs)
    return assignments, total_cost
