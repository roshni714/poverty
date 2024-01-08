from queue import PriorityQueue
import numpy as np
from tqdm import tqdm

from reporting import write_result


def solve_fractional_knapsack_problem(p_xs, convex_hulls, budget):
    """
    Priority queue algorithm of Svedrup et al 2023.
    """

    total_gain = 0
    total_spend = 0
    pq = PriorityQueue()
    assignments = {x_idx: [] for x_idx in range(len(p_xs))}
    for i in range(len(p_xs)):
        init_point = convex_hulls[i][0]
        total_spend += init_point[0] * p_xs[i]
        total_gain += init_point[1] * p_xs[i]

        if len(convex_hulls[i]) > 1:
            second_point = convex_hulls[i][1]
            ratio = (second_point[1] - init_point[1]) / (
                second_point[0] - init_point[0]
            )
            pq.put((ratio, (i, 1)))
        assignments[i] = [(init_point[1], 1)]

    while total_spend < budget and pq.qsize() > 0:
        ratio, (x_idx, hull_idx) = pq.get()
        # Remove previous assignment
        prev_spend = convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
        prev_gain = convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
        total_spend -= prev_spend
        total_gain -= prev_gain

        # Add new assignment
        curr_spend = convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx]
        curr_gain = convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx]

        total_gain += curr_gain
        total_spend += curr_spend
        assignments[x_idx] = [(convex_hulls[x_idx][hull_idx][1], 1)]

        if total_spend > budget:
            # Fractional allocation
            total_spend -= curr_spend
            total_gain -= curr_gain

            remainder = budget - total_spend

            prev_spend = convex_hulls[x_idx][hull_idx - 1][0]
            prev_gain = convex_hulls[x_idx][hull_idx - 1][1]

            curr_spend = convex_hulls[x_idx][hull_idx][0]
            curr_gain = convex_hulls[x_idx][hull_idx][1]

            c = np.minimum(
                (remainder / p_xs[x_idx] - curr_spend) / (prev_spend - curr_spend), 1.0
            )

            assignments[x_idx] = [
                (convex_hulls[x_idx][hull_idx - 1][1], c),
                (convex_hulls[x_idx][hull_idx][1], 1 - c),
            ]
            total_spend += (c * prev_spend + (1 - c) * curr_spend) * p_xs[x_idx]
            total_gain += (c * prev_gain + (1 - c) * curr_gain) * p_xs[x_idx]

            assert round(total_spend, 3) == budget

        if hull_idx != len(convex_hulls[x_idx]) - 1:
            next_hull_idx = hull_idx + 1
            next_point = convex_hulls[x_idx][next_hull_idx]
            curr_point = convex_hulls[x_idx][hull_idx]
            ratio = (next_point[1] - curr_point[1]) / (next_point[0] - curr_point[0])
            pq.put((ratio, (x_idx, next_hull_idx)))

    return assignments, total_gain


def compute_alpha_opt_policies(
    cond_dists, p_xs, budget, c_bar, n_alpha=1000, title="sim", true_cond_densities=None
):
    total_transfers = []
    opt_policies = []
    alphas = np.linspace(
        1e-3, max([dist.pdf(dist.mode) for dist in cond_dists]) - 1e-3, n_alpha
    )

    results_file = "results/{}.csv".format(title)
    for alpha in tqdm(alphas):
        cvx_hulls = [c_dist.get_convex_hull(alpha, c_bar) for c_dist in cond_dists]
        opt_policy, total_transfer = solve_fractional_knapsack_problem(
            p_xs, cvx_hulls, budget
        )

        if true_cond_densities is not None:
            prob = prob_below_line(opt_policy, c_bar, p_xs, true_cond_densities)
        else:
            prob = budget
        result = {
            "alpha": alpha,
            "total_transfer": total_transfer,
            "prob_below_line": prob,
        }
        total_transfers.append(total_transfer)
        opt_policies.append(opt_policy)
        write_result(results_file, result)
    return opt_policies, total_transfers, alphas


def prob_below_line(assignments, c_bar, p_xs, true_cond_densities):
    total_prob = 0.0
    for i in range(len(p_xs)):
        prob_below_poverty_line = 0.0
        for j in range(len(assignments[i])):
            prob_below_poverty_line += (
                true_cond_densities[i].cdf(c_bar - np.maximum(assignments[i][j][0], 0))
                * assignments[i][j][1]
            )
        total_prob += prob_below_poverty_line * p_xs[i]

    return total_prob


def policy_cost(assignments, p_xs):
    total_gap = 0
    for i in range(len(p_xs)):
        i_gap = 0.0
        for j in range(len(assignments[i])):
            i_gap += assignments[i][j][0] * assignments[i][j][1]

        total_gap += i_gap * p_xs[i]

    return total_gap
