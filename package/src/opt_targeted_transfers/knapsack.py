import numpy as np
from tqdm import tqdm
import os
import bisect
from priority_queue import PriorityQueue
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
            pq.put(ratio, (i, 1))
        assignments[i] = [(init_point[1], 1)]

    etas = [-float("inf")]
    metadata = [None]
    lamb = 0.0
    while total_spend < budget and pq:
        ratio, tups = pq.get()
        # Remove previous assignment
        prev_spend = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        prev_gain = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        total_spend -= prev_spend
        total_gain -= prev_gain

        # Add new assignment
        curr_spend = sum(
            [convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )
        curr_gain = sum(
            [convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )

        total_gain += curr_gain
        total_spend += curr_spend
        for x_idx, hull_idx in tups:
            assignments[x_idx] = [(convex_hulls[x_idx][hull_idx][1], 1)]

        etas.append(ratio)
        metadata.append(tups)
        if total_spend > budget:
            # Fractional allocation
            total_spend -= curr_spend
            total_gain -= curr_gain

            remainder = budget - total_spend

            prev_spend = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            prev_gain = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            curr_spend = sum(
                [
                    convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            curr_gain = sum(
                [
                    convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            c = np.minimum((remainder - curr_spend) / (prev_spend - curr_spend), 1.0)
            lamb = c
            for x_idx, hull_idx in tups:
                assignments[x_idx] = [
                    (convex_hulls[x_idx][hull_idx - 1][1], c),
                    (convex_hulls[x_idx][hull_idx][1], 1 - c),
                ]
            total_spend += c * prev_spend + (1 - c) * curr_spend
            total_gain += c * prev_gain + (1 - c) * curr_gain
            return assignments, total_gain, total_spend, etas[-1], lamb

        for x_idx, hull_idx in tups:
            if hull_idx != len(convex_hulls[x_idx]) - 1:
                next_hull_idx = hull_idx + 1
                next_point = convex_hulls[x_idx][next_hull_idx]
                curr_point = convex_hulls[x_idx][hull_idx]
                ratio = (next_point[1] - curr_point[1]) / (
                    next_point[0] - curr_point[0]
                )
                pq.put(ratio, (x_idx, next_hull_idx))

    return assignments, total_gain, total_spend, etas[-1], lamb


def get_transfer_function(alpha, c_bar, eta, lamb, compute_cond_density):
    def t(X_test):
        cond_densities = compute_cond_density(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}

        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cond_density.get_convex_hull(alpha, c_bar)
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


def check_assignments_are_equal(assignment1, assignment2):
    assert assignment1.keys() == assignment2.keys()

    for key in assignment1.keys():
        val1 = assignment1[key]
        val2 = assignment2[key]
        assert val1 == val2, "error at key {} bc {} != {}".format(key, val1, val2)


def compute_alpha_opt_policies(
    train_dataset,
    compute_cond_density,
    budget,
    c_bar,
    alpha_min=None,
    alpha_max=None,
    n_alpha=1000,
    path="sim"
):
    cond_dists = compute_cond_density(train_dataset.X)

    total_transfers = []
    opt_policies = []
    if alpha_max is None:
        alpha_max  = max([dist.pdf(dist.mode) for dist in cond_dists]).item()
    if alpha_min is None:
        alpha_min = max([dist.pdf(dist.mode) for dist in cond_dists]).item() / 1000
    alphas = np.linspace(min_alpha, max_alpha, n_alpha)
    print("Alpha range: {}, {}".format(alphas[0], alphas[-1]))
    results_file = path 

    if os.path.exists(results_file):
        os.remove(results_file)

    for alpha in tqdm(alphas):
        cvx_hulls = [c_dist.get_convex_hull(alpha, c_bar) for c_dist in cond_dists]
        (
            opt_assignment,
            total_transfer,
            prob_below_line,
            eta,
            lamb,
        ) = solve_fractional_knapsack_problem(train_dataset.r, cvx_hulls, budget)
        t_alpha = get_transfer_function(
            alpha, c_bar, eta, lamb, compute_cond_density=compute_cond_density
        )
        #        prox_assignment = t_alpha(train_dataset.X)
        #        check_assignments_are_equal(opt_assignment, prox_assignment)

        result = {
            "alpha": alpha,
            "total_transfer": total_transfer,
            "prob_below_line": prob_below_line,
        }
        total_transfers.append(total_transfer)
        opt_policies.append(t_alpha)
        if results_file is not None:
            write_result(results_file, result)
    return opt_policies, total_transfers, alphas
