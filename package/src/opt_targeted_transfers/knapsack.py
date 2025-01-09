import numpy as np
from tqdm import tqdm
import os
import bisect

from opt_targeted_transfers.priority_queue import PriorityQueue
from opt_targeted_transfers.reporting import write_result


def solve_fractional_mc_knapsack_problem(p_xs, convex_hulls, budget):
    """
    Priority queue algorithm of Svedrup et al 2023.

    :param p_xs: A numpy.array of weights.
    :type p_xs: list[(float, float)]
    :param convex_hulls: A list of convex hulls representing feasible solutions.
    :type convex_hulls: list[numpy.ndarray]
    :param budget: The budget.
    :type budget: float
    :return: A tuple including assignments, total_gain (total transfer amount),
             total_spend (poverty rate), threshold cost-benefit ratio, threshold probability.
    :rtype: (dict, float, float, float, float)
    """

    total_cost = 0
    total_loss = 0
    pq = PriorityQueue()
    assignments = {x_idx: [] for x_idx in range(len(p_xs))}
    for i in range(len(p_xs)):
        init_point = convex_hulls[i][0]
        total_cost += init_point[0] * p_xs[i]
        total_loss += init_point[1] * p_xs[i]

        if len(convex_hulls[i]) > 1:
            second_point = convex_hulls[i][1]
            ratio = (second_point[1] - init_point[1]) / (
                second_point[0] - init_point[0]
            )
            pq.put(ratio, (i, 1))
        assignments[i] = [(init_point[0], 1)]

    lambs = [-float("inf")]
    metadata = [None]
    lamb = 0.0
    while total_cost < budget and pq:
        ratio, tups = pq.get()
        # Remove previous assignment
        prev_cost = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        prev_loss = sum(
            [
                convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                for x_idx, hull_idx in tups
            ]
        )
        total_cost -= prev_cost
        total_loss -= prev_loss

        # Add new assignment
        curr_cost = sum(
            [convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )
        curr_loss = sum(
            [convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx] for x_idx, hull_idx in tups]
        )

        total_cost += curr_cost
        total_loss += curr_loss
        for x_idx, hull_idx in tups:
            assignments[x_idx] = [(convex_hulls[x_idx][hull_idx][0], 1)]

        lambs.append(ratio)
        metadata.append(tups)
        if total_cost > budget:
            # Fractional allocation
            total_cost -= curr_cost
            total_loss -= curr_loss

            remainder = budget - total_cost

            prev_cost = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            prev_loss = sum(
                [
                    convex_hulls[x_idx][hull_idx - 1][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            curr_cost = sum(
                [
                    convex_hulls[x_idx][hull_idx][0] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )
            curr_loss = sum(
                [
                    convex_hulls[x_idx][hull_idx][1] * p_xs[x_idx]
                    for x_idx, hull_idx in tups
                ]
            )

            eta = (remainder - curr_cost)/(prev_cost - curr_cost)
            for x_idx, hull_idx in tups:
                assignments[x_idx] = [
                    (convex_hulls[x_idx][hull_idx - 1][0], eta),
                    (convex_hulls[x_idx][hull_idx][0], 1 - eta),
                ]
            total_cost += eta * prev_cost + (1 - eta) * curr_cost
            total_loss += eta * prev_loss + (1 - eta) * curr_loss
            return assignments, total_cost, total_loss, lambs[-1], eta

        for x_idx, hull_idx in tups:
            if hull_idx != len(convex_hulls[x_idx]) - 1:
                next_hull_idx = hull_idx + 1
                next_point = convex_hulls[x_idx][next_hull_idx]
                curr_point = convex_hulls[x_idx][hull_idx]
                ratio = (next_point[1] - curr_point[1]) / (
                    next_point[0] - curr_point[0]
                )
                pq.put(ratio, (x_idx, next_hull_idx))

    return assignments, total_cost, total_loss, lambs[-1], 1.0


def get_alpha_transfer_function(
    alpha, c_bar, lamb, eta, cond_density_estimator
):
    """
    Compute the transfer function.

    :param alpha: The alpha value.
    :type alpha: float
    :param c_bar: The poverty line.
    :type c_bar: float
    :param lamb: The threshold cost-benefit ratio.
    :type lamb: float
    :param eta: The threshold probability
    :type eta: float
    :param cond_density_estimator: A function to compute the conditional density.
    :type cond_density_estimator: Callable[[np.ndarray], np.ndarray]
    :return: The transfer function.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def t(X_test):
        cond_densities = cond_density_estimator(X_test)
        assignments = {x_idx: [] for x_idx in range(len(X_test))}
        cvx_hulls = get_alpha_convex_hulls(
            alpha,
            c_bar,
            cond_dists=cond_densities,
        )
        
        for j, cond_density in enumerate(cond_densities):
            cvx_hull = cvx_hulls[j]
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
                and ratios[idx - 1] < lamb
                and ratios[idx] > lamb
            ):
                assignments[j] = [(cvx_hull[idx - 1][1], 1.0)]
            elif idx < len(ratios) and ratios[idx] == lamb:
                assignments[j] = [
                    (cvx_hull[idx - 1][1], eta),
                    (cvx_hull[idx][1], 1 - eta),
                ]
            else:
                assignments[j] = [(0., 1.0)]
        return assignments

    return t

def check_assignments_are_equal(assignment1, assignment2):
    assert assignment1.keys() == assignment2.keys()

    for key in assignment1.keys():
        val1 = assignment1[key]
        val2 = assignment2[key]
        assert val1 == val2, "error at key {} bc {} != {}".format(key, val1, val2)


def get_alpha_convex_hulls(alpha, c_bar, cond_dists):
    transfer_values = [
        c_dist.get_alpha_valid_transfers(alpha, c_bar)
        for c_dist in cond_dists
    ]

    cvx_hulls = [c_dist.get_convex_hull(z=transfer_values[i], c_bar=c_bar) for i, c_dist in enumerate(cond_dists)]
    return cvx_hulls
    

def compute_alpha_opt_policies(
    train_dataset,
    cond_density_estimator,
    budget,
    c_bar,
    min_alpha=None,
    max_alpha=None,
    n_alpha=200,
    path="sim",
):
    """
    Compute alpha-optimal policies for a given training dataset.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param cond_density_estimator: The conditional density estimator.
    :type cond_density_estimator: Callable[[np.ndarray], np.ndarray]
    :param c_bar: The poverty line.
    :type c_bar: float
    :param min_alpha: The minimum value of alpha for optimization. Defaults to None.
    :type min_alpha: float or None
    :param max_alpha: The maximum value of alpha for optimization. Defaults to None.
    :type max_alpha: float or None
    :param n_alpha: The number of alpha values to consider. Defaults to 200.
    :type n_alpha: int
    :param path: The path to save the simulation results. Defaults to "sim".
    :type path: str
    """
    cond_dists = cond_density_estimator(train_dataset.X)

    total_transfers = []
    opt_policies = []
    if max_alpha is None:
        max_alpha = np.quantile([dist.pdf(dist.mode) for dist in cond_dists], 0.97)
    if min_alpha is None:
        min_alpha = max_alpha / 1000

    alphas = np.linspace(min_alpha, max_alpha, n_alpha)
    print("Alpha range: {}, {}".format(alphas[0], alphas[-1]))
    results_file = path

    if os.path.exists(results_file):
        os.remove(results_file)

    for alpha in tqdm(alphas):
        cvx_hulls = get_alpha_convex_hulls(
            alpha=alpha,
            c_bar=c_bar,
            cond_dists=cond_dists,
        )

        (
            opt_assignment,
            total_transfer,
            prob_below_line,
            eta,
            lamb,
        ) = solve_fractional_mc_knapsack_problem(train_dataset.r, cvx_hulls, budget)
        t_alpha = get_alpha_transfer_function(
            alpha,
            c_bar,
            eta,
            lamb,
            compute_cond_density=cond_density_estimator,
        )

        # prox_assignment = t_alpha(train_dataset.X)
        # check_assignments_are_equal(opt_assignment, prox_assignment)

        result = {
            "alpha": alpha,
            "policy_cost": total_transfer,
            "poverty_rate": prob_below_line,
        }
        total_transfers.append(total_transfer)
        opt_policies.append(t_alpha)
        if results_file is not None:
            write_result(results_file, result)
    return opt_policies, total_transfers, alphas
