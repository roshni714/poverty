from queue import PriorityQueue
import numpy as np
from tqdm import tqdm
import os

from reporting import write_result
from evaluate import post_transfer_metrics, empirical_post_transfer_metrics


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

    eta = -float("inf")
    lamb = 0.0
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

        eta = ratio
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

            lamb = c
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

    return assignments, total_gain, eta, lamb


def get_transfer_function(alpha, c_bar, eta, lamb):
    def t(cond_density):
        ratios = []
        cvx_hull = cond_density.get_convex_hull(alpha, c_bar)
        for i in range(len(convex_hull) - 1):
            p1, p2 = convex_hull[i], convex_hull[i + 1]
            ratios.append((p2[1] - p1[1]) / (p2[0] - p1[0]))
        idx = bisect.bisect_right(ratios, eta) - 1
        if ratios[idx] < eta and eta < ratios[idx + 1]:
            return [(cvx_hull[idx][0], 1.0)]
        elif ratios[idx] == eta:
            return [(cvx_hull[idx][0], lamb), (cvx_hull[idx + 1][0], 1 - lamb)]

    return t


def compute_alpha_opt_policies(
    cond_dists,
    p_xs,
    budget,
    c_bar,
    n_alpha=1000,
    title="sim",
    true_cond_densities=None,
    test_dataset=None,
):
    total_transfers = []
    opt_policies = []
    max_alpha = max([dist.pdf(dist.mode) for dist in cond_dists])
    min_alpha = max([dist.pdf(dist.mode) for dist in cond_dists]) / 100
    alphas = np.linspace(min_alpha, max_alpha, n_alpha)
    print("Alpha range: {}, {}".format(min_alpha, max_alpha))
    results_file = "results/{}.csv".format(title)
    if os.path.exists(results_file):
        os.remove(results_file)
    for alpha in tqdm(alphas):
        cvx_hulls = [c_dist.get_convex_hull(alpha, c_bar) for c_dist in cond_dists]
        opt_policy, total_transfer, eta, lamb = solve_fractional_knapsack_problem(
            p_xs, cvx_hulls, budget
        )
        transfer_function = get_transfer_function(alpha, c_bar, eta, lamb)

        if true_cond_densities is not None:
            result = post_transfer_metrics(true_cond_densities, p_xs, opt_policy, c_bar)
            result["alpha"] = alpha
        elif test_dataset is not None:
            result = empirical_post_transfer_metrics(
                cond_densities, p_xs, opt_policy, c_bar
            )
            results["alpha"] = alpha

        total_transfers.append(total_transfer)
        opt_policies.append(opt_policy)
        write_result(results_file, result)
    return opt_policies, total_transfers, alphas
